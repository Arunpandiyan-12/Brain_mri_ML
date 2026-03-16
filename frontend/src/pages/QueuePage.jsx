import React, { useState, useEffect, useCallback } from "react";
import { queueAPI } from "../utils/api";
import toast from "react-hot-toast";

const SEVERITY_BADGE = {
  SEVERE: { label: "SEVERE", bg: "rgba(239,68,68,0.18)", color: "#f87171", border: "rgba(239,68,68,0.35)" },
  HIGH:   { label: "HIGH",   bg: "rgba(245,158,11,0.18)", color: "#fbbf24", border: "rgba(245,158,11,0.35)" },
  MEDIUM: { label: "MEDIUM", bg: "rgba(234,179,8,0.18)",  color: "#facc15", border: "rgba(234,179,8,0.35)"  },
  LOW:    { label: "LOW",    bg: "rgba(34,197,94,0.18)",  color: "#4ade80", border: "rgba(34,197,94,0.35)"  },
};

const URGENCY_TO_SEVERITY = {
  RED:    "SEVERE",
  YELLOW: "HIGH",
  GREEN:  "LOW",
};

const HEADACHE_MAP = ["None", "Mild", "Moderate", "Severe", "Very Severe"];

const TUMOR_LABEL = {
  glioma:     "Glioma",
  meningioma: "Meningioma",
  pituitary:  "Pituitary",
  no_tumor:   "No Tumor",
};

export default function QueuePage() {
  const [queue,    setQueue]    = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [dragIdx,  setDragIdx]  = useState(null);
  const [dragOver, setDragOver] = useState(null);

  const fetchQueue = useCallback(async () => {
    try {
      const { data } = await queueAPI.get();
      setQueue(data);
    } catch {
      toast.error("Failed to load queue");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchQueue();
    const id = setInterval(fetchQueue, 5000);
    return () => clearInterval(id);
  }, [fetchQueue]);

  const handleDragStart = (idx) => setDragIdx(idx);
  const handleDragEnter = (idx) => setDragOver(idx);

  const handleDrop = async (dropIdx) => {
    if (dragIdx === null || dragIdx === dropIdx) { setDragIdx(null); setDragOver(null); return; }
    const reordered = [...queue];
    const [item] = reordered.splice(dragIdx, 1);
    reordered.splice(dropIdx, 0, item);
    setQueue(reordered);
    setDragIdx(null);
    setDragOver(null);
    try {
      await queueAPI.reorder({ ordered_case_ids: reordered.map(c => c.case_id) });
      toast.success("Queue reordered");
    } catch {
      toast.error("Reorder failed");
      fetchQueue();
    }
  };

  return (
    <div className="h-full overflow-y-auto" style={{ background: "#0a0c18" }}>
      <div className="px-6 py-6">

        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="#00d4dc" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>
              <line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/>
              <line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>
            </svg>
            <h1 className="text-base font-bold text-white" style={{ fontFamily: "'Exo 2', sans-serif" }}>
              Priority Queue
            </h1>
          </div>
          <button
            onClick={fetchQueue}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white transition-colors"
            style={{ border: "1px solid rgba(255,255,255,0.08)", fontFamily: "'DM Sans', sans-serif" }}
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
            </svg>
            Refresh
          </button>
        </div>

        <div style={{ border: "1px solid rgba(0,212,220,0.08)", background: "rgba(6,14,30,0.85)" }}>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: "1px solid rgba(0,212,220,0.08)" }}>
                {["Rank", "Case ID", "Patient Name", "Age", "Gender", "Seizure", "Headache", "Tumor Type", "Severity"].map(col => (
                  <th
                    key={col}
                    className="text-left px-4 py-3 text-xs uppercase tracking-wider text-slate-500 font-semibold whitespace-nowrap"
                    style={{ fontFamily: "'DM Sans', sans-serif" }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} className="text-center py-16">
                    <div className="flex items-center justify-center gap-3">
                      <div
                        className="w-4 h-4 rounded-full border-2 animate-spin"
                        style={{ borderColor: "rgba(0,212,220,0.25)", borderTopColor: "#00d4dc" }}
                      />
                      <span className="text-sm text-slate-500" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                        Loading queue...
                      </span>
                    </div>
                  </td>
                </tr>
              ) : queue.length === 0 ? (
                <tr>
                  <td colSpan={9} className="text-center py-16 text-slate-600 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                    No cases in queue
                  </td>
                </tr>
              ) : (
                queue.map((item, idx) => {
                  const severityKey = URGENCY_TO_SEVERITY[item.urgency_label] ?? "LOW";
                  const sev = SEVERITY_BADGE[severityKey] ?? SEVERITY_BADGE.LOW;
                  const isDragging = dragIdx === idx;
                  const isOver     = dragOver === idx;

                  return (
                    <tr
                      key={item.case_id}
                      draggable
                      onDragStart={() => handleDragStart(idx)}
                      onDragEnter={() => handleDragEnter(idx)}
                      onDragOver={e => e.preventDefault()}
                      onDrop={() => handleDrop(idx)}
                      onDragEnd={() => { setDragIdx(null); setDragOver(null); }}
                      className="transition-all duration-150 cursor-grab active:cursor-grabbing"
                      style={{
                        borderBottom: "1px solid rgba(255,255,255,0.04)",
                        background: isDragging
                          ? "rgba(0,212,220,0.07)"
                          : isOver
                          ? "rgba(0,212,220,0.03)"
                          : "transparent",
                        opacity: isDragging ? 0.5 : 1,
                      }}
                    >
                      <td className="px-4 py-3 text-slate-500 text-xs font-mono font-bold">
                        {idx + 1}
                      </td>
                      <td className="px-4 py-3 text-xs font-mono" style={{ color: "#00d4dc" }}>
                        {item.case_id}
                      </td>
                      <td className="px-4 py-3 text-white text-sm whitespace-nowrap" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                        {item.patient_name}
                      </td>
                      <td className="px-4 py-3 text-slate-300 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                        {item.age}
                      </td>
                      <td className="px-4 py-3 text-slate-300 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                        {item.gender}
                      </td>
                      <td className="px-4 py-3 text-sm" style={{ fontFamily: "'DM Sans', sans-serif", color: item.history_seizures ? "#f87171" : "#94a3b8" }}>
                        {item.history_seizures ? "Yes" : "No"}
                      </td>
                      <td className="px-4 py-3 text-slate-300 text-sm whitespace-nowrap" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                        {typeof item.headache_severity === "number"
                          ? HEADACHE_MAP[item.headache_severity] ?? item.headache_severity
                          : item.headache_severity ?? "—"}
                      </td>
                      <td className="px-4 py-3 text-slate-300 text-sm whitespace-nowrap" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                        {item.tumor_class ? TUMOR_LABEL[item.tumor_class] ?? item.tumor_class : "—"}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className="px-2 py-0.5 text-xs font-bold"
                          style={{
                            background: sev.bg,
                            color: sev.color,
                            border: `1px solid ${sev.border}`,
                            fontFamily: "'Exo 2', sans-serif",
                          }}
                        >
                          {sev.label}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-slate-700 mt-3 text-center" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          Auto-refreshes every 5 seconds · AI reorders by urgency after each scan · Drag to manually prioritize
        </p>
      </div>
    </div>
  );
}