"""
PDF Extraction Service

Pipeline per PDF:
  1. Detect true diagram pages (heading position + graphics check)
  2. Vision LLM analyses each diagram page
  3. pdfplumber extracts all tables
  4. Smart section extraction — universal, no hardcoded section numbers:
       Keeps: headings, diagram-adjacent pages, keyword-bearing spans
       Drops: font<=8pt headers/footers, shape icons, TOC noise, irrelevant body text
  5. Write diagrams.json, tables.json, sections.json, metadata.json
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

# Keywords — any span containing these is always kept regardless of location.
# These appear in every PowerCurve document regardless of section numbering.
_RELEVANT_KEYWORDS = [
    "ws-system", "dv-application", "expflag", "pidflag",
    "securecardflag", "processflag", "tactical parameter",
    "auto approved", "auto declined", "referred",
    "new secured", "new unsecured",
    "premier bank", "pco", "bureau",
    "business process flow", "shape / component",
    "flow sequence", "abbreviation", "terminology",
    "decision", "approve", "declin",
]

FLOW_JSON_STRUCTURE = """{
  "flow_name": "clean flow name WITHOUT section numbers (e.g. 'New Application: Auto Approved Path' not 'Flow 2.3.1 — New Application: Auto Approved Path')",
  "source_system": "the submitting system (e.g. Premier Bank)",
  "target_system": "the processing system (e.g. PCO)",
  "swim_lanes": ["list of swim lane names"],
  "components": [
    {
      "name": "exact label from diagram",
      "type": "one of: start_event | task | decision | service_task | end_event",
      "swim_lane": "which lane this component is in",
      "description": "what this component does"
    }
  ],
  "relationships": [
    {
      "from": "source component name",
      "to": "target component name",
      "condition": "label on arrow if any, else empty string",
      "path_type": "one of: happy_path | exception | alternate"
    }
  ],
  "flow_paths": [
    {
      "path_name": "e.g. Auto Approved Path",
      "outcome": "one of: approved | declined | referred | unknown",
      "sequence": ["ordered component names start to end"]
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
}"""


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


def _count_flow_headings(page: fitz.Page) -> int:
    """
    Count how many 'Business Process Flow' bold headings appear on this page.

    In PowerCurve documents the structure is always:
      Section 2 Business Process Management Components
        2.3 Business Process Flows
            Business Process Flow   ← bold heading, font >= 10pt, appears once per diagram
            Flow 2.3.1 — Name
            [DIAGRAM]
            Shape/Component table

            Business Process Flow   ← next diagram starts here (can be mid-page)
            Flow 2.3.2 — Name
            [DIAGRAM]

    Multiple flows can appear on the same page — no y_fraction restriction.
    """
    count = 0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                font_size = span.get("size", 0)
                font_flags = span.get("flags", 0)
                is_bold = bool(font_flags & 2**4)
                if (
                    text == "Business Process Flow"
                    and font_size >= 10
                    and is_bold
                ):
                    count += 1
    return count


def _is_true_diagram_page(page: fitz.Page) -> bool:
    """
    A page contains diagrams if it has at least one bold 'Business Process Flow'
    heading (font >= 10pt) AND has graphical elements (drawings or images).
    No y_fraction restriction — flows can appear anywhere on the page.
    """
    if _count_flow_headings(page) == 0:
        return False
    drawings = page.get_drawings()
    images = page.get_images()
    return len(drawings) > 5 or len(images) > 0



class PDFExtractionService:

    def __init__(self, project_id: uuid.UUID):
        self.project_id = project_id
        self.source_root = settings.get_source_path() / str(project_id)
        self.pdf_dir = self.source_root / PDF_STORAGE_FOLDER
        self.artifacts_root = settings.get_artifacts_path() / str(project_id)
        self.extraction_dir = self.artifacts_root / "pdf_extraction"
        self.status_file = self.artifacts_root / "analysis_status.json"

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
                    f"{len(sections)} relevant section span(s)"
                )
            except Exception as e:
                logger.exception(f"[{self.project_id}] Failed: {pdf_path.name} -> {e}")

        await self._write_json(self.diagrams_file, all_diagrams)
        await self._write_json(self.tables_file, all_tables)
        await self._write_json(self.sections_file, all_sections)
        await self._write_json(self.metadata_file, all_metadata)

        await self._update_status(total_files, processed, "completed")
        logger.info(
            f"[{self.project_id}] Extraction complete — "
            f"{len(all_diagrams)} diagrams, "
            f"{len(all_tables)} tables, "
            f"{len(all_sections)} section spans"
        )

    async def _process_pdf(self, pdf_path: Path) -> tuple[list, list, list, dict]:
        logger.info(f"[{self.project_id}] Processing: {pdf_path.name}")
        diagrams = await self._extract_diagrams(pdf_path)
        tables = await self._extract_tables(pdf_path)
        sections = await self._extract_sections(pdf_path)
        metadata = await self._extract_metadata(pdf_path)
        return diagrams, tables, sections, metadata

    # -----------------------------------------------------------------------
    # Diagram extraction
    # -----------------------------------------------------------------------

    async def _extract_diagrams(self, pdf_path: Path) -> list[dict[str, Any]]:
        """
        Extract all Business Process Flow diagrams from the PDF.

        PowerCurve document structure:
          Section 2 Business Process Management Components
            2.3 Business Process Flows
                Business Process Flow   ← bold heading marks start of each diagram
                Flow 2.3.1 — Name       ← flow name
                [DIAGRAM]
                Shape/Component table

                Business Process Flow   ← next diagram — can be mid-page
                Flow 2.3.2 — Name

        Key fix: pages 5 and 7 each contain 2 flows. One Vision LLM call per page
        returns ALL flows on that page — the prompt explicitly asks for complete
        flow structure. We count headings per page for logging only.
        """
        if not settings.DIAGRAM_EXTRACTION_ENABLED:
            logger.info(f"[{self.project_id}] Diagram extraction disabled")
            return []

        doc = fitz.open(pdf_path)
        max_pages = min(len(doc), settings.PDF_MAX_PAGES)

        # Each entry: (page_num, flow_count_on_page, page_snippet)
        diagram_pages: list[tuple[int, int, str]] = []

        for page_num in range(max_pages):
            page = doc[page_num]
            flow_count = _count_flow_headings(page)
            if flow_count > 0:
                drawings = page.get_drawings()
                images = page.get_images()
                has_graphics = len(drawings) > 5 or len(images) > 0
                if has_graphics:
                    snippet = page.get_text("text")[:400].replace("\n", " ").strip()
                    diagram_pages.append((page_num, flow_count, snippet))
                    logger.info(
                        f"[{self.project_id}] Page {page_num + 1}: "
                        f"{flow_count} flow heading(s) detected"
                    )
                else:
                    logger.debug(
                        f"[{self.project_id}] Page {page_num + 1}: "
                        f"flow heading found but no graphics — skipping"
                    )

        if not diagram_pages:
            logger.warning(f"[{self.project_id}] No diagram pages in {pdf_path.name}")
            doc.close()
            return []

        total_flows = sum(fc for _, fc, _ in diagram_pages)
        logger.info(
            f"[{self.project_id}] {len(diagram_pages)} diagram page(s), "
            f"~{total_flows} flow(s) expected — starting Vision LLM calls"
        )

        # One Vision LLM call per page — prompt extracts ALL flows on the page
        semaphore = asyncio.Semaphore(VISION_CONCURRENCY)
        tasks = [
            asyncio.create_task(
                self._analyse_diagram_page(
                    doc[page_num], page_num, flow_count, snippet, semaphore
                )
            )
            for page_num, flow_count, snippet in diagram_pages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        diagrams = []
        for i, result in enumerate(results):
            page_num = diagram_pages[i][0]
            if isinstance(result, Exception):
                logger.error(f"[{self.project_id}] Page {page_num + 1} error: {result}")
            elif isinstance(result, list):
                # Multiple flows extracted from one page
                for flow in result:
                    if flow:
                        diagrams.append(flow)
                        logger.info(
                            f"[{self.project_id}] Page {page_num + 1}: "
                            f"'{flow.get('flow_name', 'unnamed')}' — "
                            f"{len(flow.get('components', []))} components, "
                            f"{len(flow.get('flow_paths', []))} path(s)"
                        )
            elif result is not None:
                diagrams.append(result)
                logger.info(
                    f"[{self.project_id}] Page {page_num + 1}: "
                    f"'{result.get('flow_name', 'unnamed')}' — "
                    f"{len(result.get('components', []))} components, "
                    f"{len(result.get('flow_paths', []))} path(s)"
                )
            else:
                logger.warning(f"[{self.project_id}] Page {page_num + 1}: returned None")

        doc.close()
        logger.info(
            f"[{self.project_id}] Total diagrams extracted: {len(diagrams)}"
        )
        return diagrams

    async def _analyse_diagram_page(
        self,
        page: fitz.Page,
        page_num: int,
        flow_count: int,
        page_snippet: str,
        semaphore: asyncio.Semaphore,
    ) -> list[dict[str, Any]] | None:
        """
        Send page image to Vision LLM.
        If flow_count > 1, prompt explicitly asks for all flows as a JSON array.
        Returns a list of flow dicts (even for single flows — always a list).
        """
        async with semaphore:
            try:
                zoom_matrix = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=zoom_matrix)
                img_bytes = pix.tobytes("png")
                img_b64 = _encode_image(img_bytes)

                # Build prompt based on how many flows are on this page
                if flow_count > 1:
                    prompt = (
                        f"This page contains {flow_count} separate Business Process Flow diagrams. "
                        "Extract ALL of them.\n\n"
                        "Return ONLY a valid JSON ARRAY containing one object per flow. "
                        "No markdown, no explanation, just the array:\n"
                        f"[{FLOW_JSON_STRUCTURE}, {FLOW_JSON_STRUCTURE}, ...]"
                    )
                else:
                    prompt = (
                        "This page contains one Business Process Flow diagram. "
                        "Return ONLY a valid JSON ARRAY with one object:\n"
                        f"[{FLOW_JSON_STRUCTURE}]"
                    )

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
                        {"type": "text", "text": prompt},
                    ]
                )

                response = await asyncio.to_thread(llm.invoke, [message])
                raw = response.content if hasattr(response, "content") else str(response)
                clean = _strip_json_fences(raw)

                try:
                    parsed = json.loads(clean)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[{self.project_id}] Page {page_num + 1} JSON parse error: {e}"
                        f"\nFirst 500 chars: {raw[:500]}"
                    )
                    return [{
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
                    }]

                # Normalise to list
                if isinstance(parsed, dict):
                    parsed = [parsed]

                # Tag each flow with page number
                for flow in parsed:
                    if isinstance(flow, dict):
                        flow["page"] = page_num + 1
                        flow["page_text_snippet"] = page_snippet

                return [f for f in parsed if isinstance(f, dict)]

            except Exception as e:
                logger.error(f"[{self.project_id}] Page {page_num + 1} vision failed: {e}")
                return None

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
                        f"[{self.project_id}] Page {page_num + 1} JSON parse error: {e}"
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

    # -----------------------------------------------------------------------
    # Table extraction
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Smart section extraction — only what matters for test case generation
    # -----------------------------------------------------------------------

    async def _extract_sections(self, pdf_path: Path) -> list[dict[str, Any]]:
        """
        Universal smart extraction — no hardcoded section numbers.
        Works on any PowerCurve document regardless of structure.

        KEEP:
          - All headings (font >= 10pt bold) — document structure context
          - Everything on diagram pages and adjacent pages (±1 page)
            — captures component description tables that spill across pages
          - Any span containing relevant keywords anywhere in the document
            — tactical parameters, flag names, system names, outcomes

        DROP:
          - font <= 8pt — page headers/footers (universal in PowerCurve docs)
          - spans <= 2 chars — shape icons from component tables (G, I, N)
          - TOC page body lines with no keywords
          - Everything else not matching the above
        """
        sections: list[dict[str, Any]] = []
        try:
            doc = fitz.open(pdf_path)
            max_pages = min(len(doc), settings.PDF_MAX_PAGES)

            # First pass — find diagram-adjacent pages and TOC pages
            diagram_adjacent: set[int] = set()
            toc_pages: set[int] = set()

            for page_num in range(max_pages):
                page_text = doc[page_num].get_text("text")
                if "Business Process Flow" in page_text:
                    diagram_adjacent.update([page_num - 1, page_num, page_num + 1])
                if "table of contents" in page_text.lower():
                    toc_pages.add(page_num)

            # Second pass — extract relevant spans
            for page_num in range(max_pages):
                page = doc[page_num]
                near_diagram = page_num in diagram_adjacent
                is_toc = page_num in toc_pages
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

                            # DROP: page headers/footers (font <= 8pt in all PowerCurve docs)
                            if font_size <= 8.0:
                                continue

                            # DROP: single/double char shape icons (G, I, N from tables)
                            if len(text) <= 2 and text not in ("Y", "N", "→"):
                                continue

                            # Classify span type
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

                            has_keyword = any(
                                kw in text.lower() for kw in _RELEVANT_KEYWORDS
                            )

                            # KEEP: headings always
                            if span_type in ("heading1", "heading2", "heading3"):
                                pass
                            # KEEP: diagram pages and adjacent — everything
                            elif near_diagram:
                                pass
                            # KEEP: keyword spans anywhere
                            elif has_keyword:
                                pass
                            # DROP: TOC body lines without keywords
                            elif is_toc:
                                continue
                            # DROP: everything else
                            else:
                                continue

                            sections.append({
                                "page": page_num + 1,
                                "text": text,
                                "type": span_type,
                                "font_size": round(font_size, 1),
                                "is_bold": is_bold,
                            })

            doc.close()
            logger.info(
                f"[{self.project_id}] Smart extraction: "
                f"{len(sections)} relevant spans from {pdf_path.name} "
                f"(diagram-adjacent: {sorted(p + 1 for p in diagram_adjacent if 0 <= p < max_pages)})"
            )
        except Exception as e:
            logger.error(f"[{self.project_id}] Section extraction failed: {e}")
        return sections

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

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
