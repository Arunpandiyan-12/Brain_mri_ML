"""
PDF Extraction Service

Pipeline per PDF:
  1. Detect true diagram pages (heading position + graphics check)
  2. Vision LLM analyses each diagram — full flow structure
  3. pdfplumber extracts all tables (component descriptions, flag combos)
  4. PyMuPDF extracts all text spans with type classification
  5. Write diagrams.json, tables.json, sections.json, metadata.json
  6. Write per-flow .md files to app/data/{project_id}/ for RAG indexing
     One .md per flow containing:
       - Flow structure (components, decision nodes, paths)
       - Component description table (Shape | Description)
       - Relevant tactical parameters from sections
"""

import asyncio
import base64
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import fitz
import pdfplumber
from langchain_core.messages import HumanMessage
from loguru import logger

from app.config.llm_config import get_llm
from app.config.settings import settings

VISION_CONCURRENCY = 3
PDF_STORAGE_FOLDER = "pdf_ingested/raw"

_MIN_HEADING_Y_FRACTION = 0.05
_MAX_HEADING_Y_FRACTION = 0.35

DIAGRAM_PROMPT = """\
You are an expert business process analyst parsing a PowerCurve Business Process Flow diagram.

Analyse this diagram image carefully and extract the COMPLETE flow structure including:
- Every component/shape (start events, tasks, decision gateways, end events)
- The swim lane each component belongs to
- Every relationship/arrow between components
- The CONDITION on each arrow leaving a decision diamond (e.g. "Y", "N", "Pass", "Fail", "Refer")
- ALL paths through the diagram — happy path AND every exception/alternate branch
- Which path leads to each terminal outcome (Auto Approved / Auto Declined / Referred)

Return ONLY a valid JSON object with exactly this structure — no markdown, no explanation:
{
  "flow_name": "name of this business process flow",
  "source_system": "the submitting system (e.g. Premier Bank)",
  "target_system": "the processing system (e.g. PCO)",
  "swim_lanes": ["list of swim lane names found in diagram"],
  "components": [
    {
      "name": "exact label from diagram",
      "type": "one of: start_event | task | decision | service_task | end_event",
      "swim_lane": "which lane this component is in",
      "description": "what this component does based on its label and position"
    }
  ],
  "relationships": [
    {
      "from": "source component name",
      "to": "target component name",
      "condition": "label on the arrow if any, else empty string",
      "path_type": "one of: happy_path | exception | alternate"
    }
  ],
  "flow_paths": [
    {
      "path_name": "e.g. Auto Approved Path",
      "outcome": "one of: approved | declined | referred | unknown",
      "sequence": ["ordered list of component names from start to end"]
    }
  ],
  "decision_nodes": [
    {
      "name": "decision component name",
      "condition_description": "what is being decided",
      "branches": [
        {"condition": "Y", "leads_to": "next component name"},
        {"condition": "N", "leads_to": "next component name"}
      ]
    }
  ]
}

If you cannot determine a value, use empty string or empty list. Never omit a key.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_image(img_bytes: bytes) -> str:
    return base64.standard_b64encode(img_bytes).decode("utf-8")


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[1:end]).strip()
    return text


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[\s]+", "_", name).lower()
    return name[:80] or "flow"


def _is_true_diagram_page(page: fitz.Page) -> bool:
    page_height = page.rect.height
    text_dict = page.get_text("dict")
    found_heading = False

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if "Business Process Flow" not in text:
                    continue
                y0 = span.get("origin", [0, 0])[1]
                font_size = span.get("size", 0)
                y_fraction = y0 / page_height if page_height > 0 else 0
                if (
                    _MIN_HEADING_Y_FRACTION <= y_fraction <= _MAX_HEADING_Y_FRACTION
                    and font_size >= 10
                ):
                    found_heading = True
                    break
            if found_heading:
                break
        if found_heading:
            break

    if not found_heading:
        return False

    drawings = page.get_drawings()
    images = page.get_images()
    return len(drawings) > 5 or len(images) > 0


def _build_flow_markdown(
    diagram: dict[str, Any],
    component_table_rows: list[dict],
    context_sections: list[dict],
) -> str:
    """
    Build a rich .md file for one flow containing:
    - Flow metadata (name, source, target, swim lanes)
    - Decision nodes with all branches
    - All flow paths (approved/declined/referred)
    - Component description table (Shape | Description)
    - Relevant context (tactical parameters, abbreviations)
    """
    lines = []

    flow_name = diagram.get("flow_name", "Unknown Flow")
    source = diagram.get("source_system", "")
    target = diagram.get("target_system", "")
    lanes = diagram.get("swim_lanes", [])

    lines += [
        f"# {flow_name}",
        "",
        f"**Source System:** {source}",
        f"**Target System:** {target}",
        f"**Swim Lanes:** {', '.join(lanes)}",
        "",
    ]

    # Flow paths
    flow_paths = diagram.get("flow_paths", [])
    if flow_paths:
        lines += ["## Flow Paths", ""]
        for path in flow_paths:
            lines += [
                f"### {path.get('path_name', 'Unnamed Path')}",
                f"**Outcome:** {path.get('outcome', '')}",
                f"**Sequence:** {' → '.join(path.get('sequence', []))}",
                "",
            ]

    # Decision nodes
    decision_nodes = diagram.get("decision_nodes", [])
    if decision_nodes:
        lines += ["## Decision Nodes", ""]
        for node in decision_nodes:
            lines += [
                f"### {node.get('name', '')}",
                f"**Condition:** {node.get('condition_description', '')}",
                "**Branches:**",
            ]
            for branch in node.get("branches", []):
                lines.append(
                    f"- If `{branch.get('condition', '?')}` → {branch.get('leads_to', '?')}"
                )
            lines.append("")

    # Relationships
    relationships = diagram.get("relationships", [])
    if relationships:
        lines += ["## Relationships", ""]
        for rel in relationships:
            cond = f" [{rel.get('condition')}]" if rel.get("condition") else ""
            lines.append(
                f"- {rel.get('from', '')} →{cond} {rel.get('to', '')} "
                f"({rel.get('path_type', '')})"
            )
        lines.append("")

    # Component description table (from pdfplumber)
    if component_table_rows:
        lines += ["## Component Descriptions", ""]
        lines += ["| Shape / Component | Description |", "|---|---|"]
        for row in component_table_rows:
            shape = list(row.values())[0] if row else ""
            desc = list(row.values())[1] if len(row) > 1 else ""
            shape = str(shape).replace("|", "\\|").replace("\n", " ")
            desc = str(desc).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {shape} | {desc} |")
        lines.append("")

    # All components from diagram
    components = diagram.get("components", [])
    if components:
        lines += ["## Components", ""]
        for comp in components:
            lines += [
                f"### {comp.get('name', '')}",
                f"**Type:** {comp.get('type', '')}",
                f"**Swim Lane:** {comp.get('swim_lane', '')}",
                f"**Description:** {comp.get('description', '')}",
                "",
            ]

    # Context sections (tactical parameters, abbreviations, connectivity)
    if context_sections:
        lines += ["## Context", ""]
        for section in context_sections:
            lines.append(section.get("text", ""))
        lines.append("")

    return "\n".join(lines)


class PDFExtractionService:

    def __init__(self, project_id: uuid.UUID):
        self.project_id = project_id
        self.source_root = settings.get_source_path() / str(project_id)
        self.pdf_dir = self.source_root / PDF_STORAGE_FOLDER
        self.artifacts_root = settings.get_artifacts_path() / str(project_id)
        self.extraction_dir = self.artifacts_root / "pdf_extraction"
        self.status_file = self.artifacts_root / "analysis_status.json"
        self.rag_data_dir = Path(settings.RAG_DATA_PATH) / str(project_id)

        self.diagrams_file = self.extraction_dir / "diagrams.json"
        self.tables_file = self.extraction_dir / "tables.json"
        self.sections_file = self.extraction_dir / "sections.json"
        self.metadata_file = self.extraction_dir / "metadata.json"

    async def get_status(self) -> dict:
        if not self.status_file.exists():
            return {
                "status": "not_started",
                "started_at": None,
                "completed_at": None,
                "files_total": 0,
                "files_processed": 0,
                "error": None,
                "steps": [],
            }
        async with aiofiles.open(self.status_file, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    async def run(self) -> None:
        pdf_files = (
            list(sorted(self.pdf_dir.glob("*.pdf")))
            if self.pdf_dir.exists()
            else []
        )
        total_files = len(pdf_files)

        if total_files == 0:
            logger.warning(f"[{self.project_id}] No PDFs found in {self.pdf_dir}")
            await self._update_status(0, 0, "completed", "No PDFs found")
            return

        await self._update_status(total_files, 0, "in_progress", None)
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        self.rag_data_dir.mkdir(parents=True, exist_ok=True)

        all_diagrams: list[dict[str, Any]] = []
        all_tables: list[dict[str, Any]] = []
        all_sections: list[dict[str, Any]] = []
        all_metadata: list[dict[str, Any]] = []

        processed = 0
        for pdf_path in pdf_files:
            try:
                diagrams, tables, sections, metadata = await self._process_pdf(pdf_path)
                all_diagrams.extend(diagrams)
                all_tables.extend(tables)
                all_sections.extend(sections)
                all_metadata.append(metadata)
                processed += 1
                await self._update_status(total_files, processed, "in_progress")
                logger.info(
                    f"[{self.project_id}] {pdf_path.name}: "
                    f"{len(diagrams)} diagram(s), "
                    f"{len(tables)} table(s), "
                    f"{len(sections)} section(s)"
                )
            except Exception as e:
                logger.exception(f"[{self.project_id}] Failed: {pdf_path.name} -> {e}")

        await self._write_json(self.diagrams_file, all_diagrams)
        await self._write_json(self.tables_file, all_tables)
        await self._write_json(self.sections_file, all_sections)
        await self._write_json(self.metadata_file, all_metadata)

        self._write_flow_markdown_files(all_diagrams, all_tables, all_sections)

        await self._update_status(total_files, processed, "completed")
        logger.info(
            f"[{self.project_id}] Extraction complete — "
            f"{len(all_diagrams)} diagrams, "
            f"{len(all_tables)} tables, "
            f"{len(all_sections)} text spans"
        )

    async def _process_pdf(self, pdf_path: Path) -> tuple[list, list, list, dict]:
        logger.info(f"[{self.project_id}] Processing: {pdf_path.name}")
        diagrams = await self._extract_diagrams(pdf_path)
        tables = await self._extract_tables(pdf_path)
        sections = await self._extract_sections(pdf_path)
        metadata = await self._extract_metadata(pdf_path)
        return diagrams, tables, sections, metadata

    async def _extract_diagrams(self, pdf_path: Path) -> list[dict[str, Any]]:
        if not settings.DIAGRAM_EXTRACTION_ENABLED:
            logger.info(f"[{self.project_id}] Diagram extraction disabled")
            return []

        doc = fitz.open(pdf_path)
        max_pages = min(len(doc), settings.PDF_MAX_PAGES)
        diagram_pages: list[tuple[int, str]] = []

        for page_num in range(max_pages):
            page = doc[page_num]
            if _is_true_diagram_page(page):
                snippet = page.get_text("text")[:300].replace("\n", " ").strip()
                diagram_pages.append((page_num, snippet))
                logger.debug(f"[{self.project_id}] Diagram page confirmed: {page_num + 1}")
            else:
                if "Business Process Flow" in page.get_text("text"):
                    logger.debug(
                        f"[{self.project_id}] Page {page_num + 1} skipped — "
                        "text match but failed heading/graphics check"
                    )

        if not diagram_pages:
            logger.warning(f"[{self.project_id}] No diagram pages in {pdf_path.name}")
            doc.close()
            return []

        logger.info(
            f"[{self.project_id}] {len(diagram_pages)} diagram page(s) — "
            "starting Vision LLM calls"
        )

        semaphore = asyncio.Semaphore(VISION_CONCURRENCY)
        tasks = [
            asyncio.create_task(
                self._analyse_diagram_page(doc[page_num], page_num, snippet, semaphore)
            )
            for page_num, snippet in diagram_pages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        diagrams = []
        for i, result in enumerate(results):
            page_num = diagram_pages[i][0]
            if isinstance(result, Exception):
                logger.error(f"[{self.project_id}] Page {page_num + 1} error: {result}")
            elif result is not None:
                diagrams.append(result)
                logger.info(
                    f"[{self.project_id}] Page {page_num + 1}: "
                    f"'{result.get('flow_name', 'unnamed')}' — "
                    f"{len(result.get('components', []))} components, "
                    f"{len(result.get('flow_paths', []))} path(s), "
                    f"{len(result.get('decision_nodes', []))} decision(s)"
                )
            else:
                logger.warning(f"[{self.project_id}] Page {page_num + 1}: returned None")

        doc.close()
        return diagrams

    async def _analyse_diagram_page(
        self,
        page: fitz.Page,
        page_num: int,
        page_snippet: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any] | None:
        async with semaphore:
            try:
                zoom_matrix = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=zoom_matrix)
                img_bytes = pix.tobytes("png")
                img_b64 = _encode_image(img_bytes)

                llm = get_llm()
                message = HumanMessage(
                    content=[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": DIAGRAM_PROMPT},
                    ]
                )

                response = await asyncio.to_thread(llm.invoke, [message])
                raw = response.content if hasattr(response, "content") else str(response)
                clean = _strip_json_fences(raw)

                try:
                    parsed = json.loads(clean)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[{self.project_id}] Page {page_num + 1} parse error: {e}"
                        f"\nFirst 500 chars: {raw[:500]}"
                    )
                    return {
                        "page": page_num + 1,
                        "flow_name": f"Page {page_num + 1} — parse failed",
                        "parse_error": str(e),
                        "raw_response": raw,
                        "components": [],
                        "relationships": [],
                        "flow_paths": [],
                        "decision_nodes": [],
                        "swim_lanes": [],
                        "source_system": "",
                        "target_system": "",
                    }

                parsed["page"] = page_num + 1
                parsed["page_text_snippet"] = page_snippet
                return parsed

            except Exception as e:
                logger.error(f"[{self.project_id}] Page {page_num + 1} vision failed: {e}")
                return None

    async def _extract_tables(self, pdf_path: Path) -> list[dict[str, Any]]:
        tables: list[dict[str, Any]] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                max_pages = min(len(pdf.pages), settings.PDF_MAX_PAGES)
                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text() or ""
                    near_flow = "Business Process Flow" in page_text

                    for table_idx, table in enumerate(page.extract_tables() or []):
                        if not table or len(table) < 2:
                            continue

                        headers = [
                            str(h).strip() if h else f"col_{i}"
                            for i, h in enumerate(table[0])
                        ]
                        rows = []
                        for row in table[1:]:
                            row_dict = {
                                headers[i]: str(cell).strip() if cell else ""
                                for i, cell in enumerate(row)
                                if i < len(headers)
                            }
                            if any(v for v in row_dict.values()):
                                rows.append(row_dict)

                        if not rows:
                            continue

                        tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "near_flow_diagram": near_flow,
                            "headers": headers,
                            "rows": rows,
                            "row_count": len(rows),
                        })

            logger.info(f"[{self.project_id}] {len(tables)} table(s) from {pdf_path.name}")
        except Exception as e:
            logger.error(f"[{self.project_id}] Table extraction failed: {e}")
        return tables

    async def _extract_sections(self, pdf_path: Path) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        try:
            doc = fitz.open(pdf_path)
            max_pages = min(len(doc), settings.PDF_MAX_PAGES)

            for page_num in range(max_pages):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue

                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)
                            is_bold = bool(font_flags & 2**4)

                            if font_size >= 13:
                                span_type = "heading1"
                            elif font_size >= 11 and is_bold:
                                span_type = "heading2"
                            elif font_size >= 10 and is_bold:
                                span_type = "heading3"
                            elif "→" in text or "->" in text:
                                span_type = "flow_sequence"
                            else:
                                span_type = "body"

                            sections.append({
                                "page": page_num + 1,
                                "text": text,
                                "type": span_type,
                                "font_size": round(font_size, 1),
                                "is_bold": is_bold,
                            })

            doc.close()
            logger.info(f"[{self.project_id}] {len(sections)} text spans from {pdf_path.name}")
        except Exception as e:
            logger.error(f"[{self.project_id}] Section extraction failed: {e}")
        return sections

    async def _extract_metadata(self, pdf_path: Path) -> dict[str, Any]:
        meta: dict[str, Any] = {"filename": ""}
        try:
            meta["filename"] = pdf_path.name
            doc = fitz.open(pdf_path)
            meta.update(doc.metadata or {})
            meta["page_count"] = len(doc)
            doc.close()
        except Exception as e:
            logger.error(f"[{self.project_id}] Metadata extraction failed: {e}")
        return meta

    def _write_flow_markdown_files(
        self,
        diagrams: list[dict[str, Any]],
        tables: list[dict[str, Any]],
        sections: list[dict[str, Any]],
    ) -> None:
        """
        Write one .md file per flow to app/data/{project_id}/.
        Also write a context.md with abbreviations, tactical parameters,
        connectivity details from non-diagram sections.
        """
        if not diagrams:
            logger.warning(f"[{self.project_id}] No diagrams — skipping .md file writing")
            return

        # Build page-to-tables lookup
        tables_by_page: dict[int, list[dict]] = {}
        for t in tables:
            page = t.get("page", 0)
            tables_by_page.setdefault(page, []).append(t)

        # Context sections (abbreviations, tactical parameters, connectivity)
        context_keywords = [
            "tactical parameter", "abbreviation", "terminology",
            "connectivity", "authentication", "enrichment", "bureau",
            "expflag", "pidflag", "securecardflag", "processflag",
        ]
        context_sections = [
            s for s in sections
            if any(kw in s.get("text", "").lower() for kw in context_keywords)
        ]

        written = 0
        for diagram in diagrams:
            if diagram.get("parse_error"):
                logger.warning(
                    f"[{self.project_id}] Skipping .md for page "
                    f"{diagram.get('page')} — parse failed"
                )
                continue

            flow_name = diagram.get("flow_name", "Unknown Flow")
            page = diagram.get("page", 0)

            # Get component table rows from the same page or adjacent pages
            component_rows = []
            for p in [page, page + 1, page - 1]:
                page_tables = tables_by_page.get(p, [])
                for t in page_tables:
                    headers = t.get("headers", [])
                    if len(headers) >= 2:
                        # Component tables have Shape/Component + Description headers
                        h0 = str(headers[0]).lower()
                        h1 = str(headers[1]).lower()
                        if any(w in h0 for w in ["shape", "component", "name"]) or \
                           any(w in h1 for w in ["description", "desc"]):
                            component_rows.extend(t.get("rows", []))

            md_content = _build_flow_markdown(diagram, component_rows, context_sections)
            safe_name = _safe_filename(flow_name)
            md_path = self.rag_data_dir / f"{safe_name}.md"

            try:
                md_path.write_text(md_content, encoding="utf-8")
                written += 1
                logger.info(
                    f"[{self.project_id}] Written flow .md: {md_path.name} "
                    f"({len(md_content)} chars)"
                )
            except Exception as e:
                logger.error(f"[{self.project_id}] Failed to write {md_path}: {e}")

        # Write combined context .md
        if context_sections:
            try:
                ctx_lines = ["# Document Context\n"]
                ctx_lines += [s.get("text", "") for s in context_sections]
                ctx_path = self.rag_data_dir / "context.md"
                ctx_path.write_text("\n".join(ctx_lines), encoding="utf-8")
                logger.info(f"[{self.project_id}] Written context.md")
            except Exception as e:
                logger.error(f"[{self.project_id}] Failed to write context.md: {e}")

        logger.info(
            f"[{self.project_id}] .md files written: {written} flows → {self.rag_data_dir}"
        )

    async def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, default=str))
        tmp.replace(path)

    async def _update_status(
        self,
        total: int,
        processed: int,
        status: str = "in_progress",
        error: str | None = None,
    ) -> None:
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        current = await self.get_status()
        steps = current.get("steps", [])
        for step in steps:
            if step.get("name") == "pdf_extraction":
                step["status"] = status
                step["completed_at"] = _now() if status != "in_progress" else None

        data = {
            **current,
            "status": status,
            "steps": steps,
            "files_total": total,
            "files_processed": processed,
            "error": error,
        }
        tmp = self.status_file.with_suffix(".json.tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, default=str))
        tmp.replace(self.status_file)
