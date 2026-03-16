import React, { useEffect, useState } from "react";
import { casesAPI } from "../utils/api";

const HEADACHE_MAP = ["None", "Mild", "Moderate", "Severe", "Very Severe"];

export default function PatientsInfoPage() {
  const [cases,   setCases]   = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    casesAPI.list()
      .then(res => setCases(res.data))
      .catch(err => console.error("Failed to load cases", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div
        className="h-full flex items-center justify-center gap-3"
        style={{ background: "#0a0c18" }}
      >
        <div
          className="w-4 h-4 rounded-full border-2 animate-spin"
          style={{ borderColor: "rgba(0,212,220,0.25)", borderTopColor: "#00d4dc" }}
        />
        <span className="text-sm text-slate-500" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          Loading patients...
        </span>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto" style={{ background: "#0a0c18" }}>
      <div className="px-6 py-6">

        <div className="flex items-center gap-2 mb-5">
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="#00d4dc" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/>
            <path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
          </svg>
          <div>
            <h1 className="text-base font-bold text-white" style={{ fontFamily: "'Exo 2', sans-serif" }}>
              Patient Information
            </h1>
          </div>
        </div>

        <div style={{ border: "1px solid rgba(0,212,220,0.08)", background: "rgba(6,14,30,0.85)" }}>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: "1px solid rgba(0,212,220,0.08)" }}>
                {["Case ID", "Patient Name", "Age", "Gender", "Seizure", "Headache"].map(col => (
                  <th
                    key={col}
                    className="text-left px-5 py-3 text-xs uppercase tracking-wider text-slate-500 font-semibold"
                    style={{ fontFamily: "'DM Sans', sans-serif" }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cases.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-slate-600 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                    No patient records found
                  </td>
                </tr>
              ) : (
                cases.map((c) => (
                  <tr
                    key={c.case_id}
                    className="transition-colors hover:bg-white/[0.025]"
                    style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
                  >
                    <td className="px-5 py-3 font-medium text-xs font-mono" style={{ color: "#00d4dc" }}>
                      {c.case_id}
                    </td>
                    <td className="px-5 py-3 text-white text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                      {c.patient_name}
                    </td>
                    <td className="px-5 py-3 text-slate-300 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                      {c.age}
                    </td>
                    <td className="px-5 py-3 text-slate-300 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                      {c.gender}
                    </td>
                    <td className="px-5 py-3">
                      <span
                        className="px-2.5 py-0.5 text-xs font-semibold"
                        style={{
                          background: c.history_seizures ? "rgba(239,68,68,0.15)" : "rgba(34,197,94,0.15)",
                          color:      c.history_seizures ? "#f87171" : "#4ade80",
                          fontFamily: "'DM Sans', sans-serif",
                        }}
                      >
                        {c.history_seizures ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-slate-300 text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                      {typeof c.headache_severity === "number"
                        ? HEADACHE_MAP[c.headache_severity] ?? c.headache_severity
                        : c.headache_severity ?? "—"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-slate-700 mt-3 text-center" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          {cases.length} record{cases.length !== 1 ? "s" : ""} · Brain MRI Case Registry
        </p>
      </div>
    </div>
  );
}