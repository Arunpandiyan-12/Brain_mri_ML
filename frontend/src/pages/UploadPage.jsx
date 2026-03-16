import React, { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { casesAPI, scanAPI, reportAPI } from "../utils/api";

const HEADACHE_OPTIONS = ["None", "Mild", "Moderate", "Severe", "Very Severe"];

const TUMOR_LABELS = {
  glioma:     { label: "Glioma",     color: "#ef4444" },
  meningioma: { label: "Meningioma", color: "#f97316" },
  pituitary:  { label: "Pituitary",  color: "#38bdf8" },
  no_tumor:   { label: "No Tumor",   color: "#22c55e" },
};

const URGENCY_CONFIG = {
  RED:    { label: "Critical — Immediate Review",  color: "#ef4444", bg: "rgba(239,68,68,0.1)",    border: "rgba(239,68,68,0.3)"    },
  YELLOW: { label: "Moderate — Priority Review",   color: "#f59e0b", bg: "rgba(245,158,11,0.1)",   border: "rgba(245,158,11,0.3)"   },
  GREEN:  { label: "Low Risk — Routine Review",    color: "#22c55e", bg: "rgba(34,197,94,0.1)",    border: "rgba(34,197,94,0.3)"    },
};

export default function UploadPage() {
  const [file,        setFile]        = useState(null);
  const [preview,     setPreview]     = useState(null);
  const [existingPatients, setExistingPatients] = useState([]);
  const [form, setForm] = useState({
    case_id:           `CASE-${Date.now().toString(36).toUpperCase()}`,
    patient_name:      "",
    age:               "",
    gender:            "Male",
    headache_severity: 0,
    history_seizures:  false,
    er_admission:      false,
  });
  const [step,        setStep]        = useState("idle");
  const [result,      setResult]      = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    casesAPI.list().then(r => setExistingPatients(r.data || [])).catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  const onDrop = useCallback((accepted) => {
    if (!accepted[0]) return;
    setFile(accepted[0]);
    setPreview(URL.createObjectURL(accepted[0]));
    setResult(null);
    setStep("idle");
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp"] },
    maxFiles: 1,
  });

  const handleSelectExisting = (e) => {
    const caseId = e.target.value;
    if (!caseId) return;
    const patient = existingPatients.find(p => p.case_id === caseId);
    if (patient) {
      setForm(f => ({
        ...f,
        case_id:          patient.case_id,
        patient_name:     patient.patient_name,
        age:              patient.age,
        gender:           patient.gender,
        history_seizures: patient.history_seizures ?? false,
        headache_severity: patient.headache_severity ?? 0,
      }));
    }
  };

  const handleScan = async () => {
    if (!file)              return toast.error("Upload an MRI image first");
    if (!form.patient_name) return toast.error("Patient name is required");
    if (!form.age)          return toast.error("Patient age is required");

    clearInterval(pollRef.current);
    setResult(null);

    try {
      setStep("uploading");

      const { data: caseData } = await casesAPI.create({
        ...form,
        age:               parseInt(form.age),
        headache_severity: parseInt(form.headache_severity),
      });

      const fd = new FormData();
      fd.append("file", file);
      await scanAPI.upload(caseData.case_id, fd);

      setStep("analyzing");
      await scanAPI.analyze(caseData.case_id);

      const thisCaseId = caseData.case_id;
      let attempts = 0;

      pollRef.current = setInterval(async () => {
        attempts++;
        try {
          const { data: status } = await scanAPI.status(thisCaseId);
          if (status.status === "done") {
            clearInterval(pollRef.current);
            const { data: res } = await scanAPI.result(thisCaseId);
            setResult({ ...res, case_id: thisCaseId });
            setStep("done");
            toast.success(`Analysis complete: ${TUMOR_LABELS[res.tumor_class]?.label ?? res.tumor_class}`);
          } else if (status.status === "error") {
            clearInterval(pollRef.current);
            setStep("error");
            toast.error("Analysis failed. Please try again.");
          }
        } catch {}
        if (attempts >= 30) {
          clearInterval(pollRef.current);
          setStep("error");
          toast.error("Analysis timed out");
        }
      }, 2000);

    } catch (err) {
      setStep("error");
      toast.error(err?.response?.data?.detail ?? "Something went wrong");
    }
  };

  const isProcessing = step === "uploading" || step === "analyzing";

  return (
    <div className="h-full overflow-y-auto" style={{ background: "#0a0c18" }}>
      <div className="max-w-2xl mx-auto px-6 py-6">

        <div className="flex items-center gap-2 mb-5">
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" style={{ color: "#00d4dc" }}>
            <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
            <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/>
          </svg>
          <h1 className="text-base font-bold text-white" style={{ fontFamily: "'Exo 2', sans-serif" }}>
            Upload MRI Image
          </h1>
        </div>

        <Section title="Patient Information">
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                Select Existing Patient
              </label>
              <div className="flex gap-2">
                <select
                  onChange={handleSelectExisting}
                  defaultValue=""
                  className="flex-1 px-3 py-2 text-sm text-slate-300 outline-none"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    fontFamily: "'DM Sans', sans-serif",
                  }}
                >
                  <option value="" style={{ background: "#0f1629" }}>Choose patient or create new</option>
                  {existingPatients.map(p => (
                    <option key={p.case_id} value={p.case_id} style={{ background: "#0f1629" }}>
                      {p.patient_name} — {p.case_id}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => setForm(f => ({
                    ...f,
                    case_id: `CASE-${Date.now().toString(36).toUpperCase()}`,
                    patient_name: "", age: "", gender: "Male",
                    headache_severity: 0, history_seizures: false, er_admission: false,
                  }))}
                  className="px-3 py-2 text-xs font-semibold transition-all"
                  style={{
                    background: "rgba(0,212,220,0.12)",
                    border: "1px solid rgba(0,212,220,0.3)",
                    color: "#00d4dc",
                    fontFamily: "'DM Sans', sans-serif",
                  }}
                >
                  New Patient
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Case ID">
                <input
                  value={form.case_id}
                  readOnly
                  className="w-full px-3 py-2 text-xs text-slate-500 font-mono outline-none cursor-default"
                  style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}
                />
              </Field>
              <Field label="Patient Name">
                <input
                  value={form.patient_name}
                  onChange={e => setForm(f => ({ ...f, patient_name: e.target.value }))}
                  placeholder="John"
                  className="w-full px-3 py-2 text-sm text-white outline-none transition-colors"
                  style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", fontFamily: "'DM Sans', sans-serif" }}
                  onFocus={e => (e.target.style.borderColor = "rgba(0,212,220,0.4)")}
                  onBlur={e => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
                />
              </Field>
              <Field label="Age">
                <input
                  type="number" min="0" max="120"
                  value={form.age}
                  onChange={e => setForm(f => ({ ...f, age: e.target.value }))}
                  placeholder="60"
                  className="w-full px-3 py-2 text-sm text-white outline-none transition-colors"
                  style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", fontFamily: "'DM Sans', sans-serif" }}
                  onFocus={e => (e.target.style.borderColor = "rgba(0,212,220,0.4)")}
                  onBlur={e => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
                />
              </Field>
              <Field label="Gender">
                <select
                  value={form.gender}
                  onChange={e => setForm(f => ({ ...f, gender: e.target.value }))}
                  className="w-full px-3 py-2 text-sm text-white outline-none"
                  style={{ background: "#0f1629", border: "1px solid rgba(255,255,255,0.1)", fontFamily: "'DM Sans', sans-serif" }}
                >
                  {["Male", "Female", "Other"].map(g => (
                    <option key={g} value={g} style={{ background: "#0f1629" }}>{g}</option>
                  ))}
                </select>
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-3 items-end">
              <Field label="Headache Severity">
                <select
                  value={form.headache_severity}
                  onChange={e => setForm(f => ({ ...f, headache_severity: Number(e.target.value) }))}
                  className="w-full px-3 py-2 text-sm text-white outline-none"
                  style={{ background: "#0f1629", border: "1px solid rgba(255,255,255,0.1)", fontFamily: "'DM Sans', sans-serif" }}
                >
                  {HEADACHE_OPTIONS.map((opt, i) => (
                    <option key={i} value={i} style={{ background: "#0f1629" }}>{opt}</option>
                  ))}
                </select>
              </Field>

              <div className="flex items-center gap-2 pb-0.5">
                <input
                  type="checkbox"
                  id="seizures"
                  checked={form.history_seizures}
                  onChange={e => setForm(f => ({ ...f, history_seizures: e.target.checked }))}
                  className="w-4 h-4 cursor-pointer accent-cyan-400"
                />
                <label
                  htmlFor="seizures"
                  className="text-sm text-slate-300 cursor-pointer select-none"
                  style={{ fontFamily: "'DM Sans', sans-serif" }}
                >
                  History of Seizures
                </label>
              </div>
            </div>
          </div>
        </Section>

        <Section title="MRI Image Upload">
          <div
            {...getRootProps()}
            className="cursor-pointer transition-colors duration-200"
            style={{
              border: `2px dashed ${isDragActive ? "#00d4dc" : "rgba(255,255,255,0.1)"}`,
              background: isDragActive ? "rgba(0,212,220,0.04)" : "rgba(255,255,255,0.01)",
              minHeight: 110,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "20px",
            }}
          >
            <input {...getInputProps()} />
            {!preview ? (
              <>
                <svg className="mb-2" width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
                  <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/>
                </svg>
                <p className="text-sm text-slate-500" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                  Drag & drop MRI images here or click to browse
                </p>
                <p className="text-xs text-slate-700 mt-1" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                  JPG/PNG only
                </p>
              </>
            ) : null}
          </div>

          {preview && (
            <div
              className="mt-3 flex items-center gap-3 px-3 py-2"
              style={{ background: "rgba(0,212,220,0.06)", border: "1px solid rgba(0,212,220,0.2)" }}
            >
              <div
                className="text-xs font-medium px-2 py-0.5"
                style={{ background: "rgba(0,212,220,0.15)", color: "#00d4dc", fontFamily: "'DM Sans', sans-serif" }}
              >
                ✓ 1 image selected
              </div>
              <img src={preview} alt="preview" className="w-12 h-12 object-cover" style={{ border: "1px solid rgba(255,255,255,0.1)" }} />
              <div>
                <p className="text-xs text-slate-300" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                  {file?.name}
                </p>
                <p className="text-xs text-slate-600" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                  {file ? `${(file.size / 1024).toFixed(0)} KB` : ""}
                </p>
              </div>
            </div>
          )}

          {step === "analyzing" && (
            <div
              className="mt-2 flex items-center gap-2 px-3 py-2 text-xs"
              style={{ background: "rgba(0,212,220,0.05)", border: "1px solid rgba(0,212,220,0.15)", color: "#00d4dc", fontFamily: "'DM Sans', sans-serif" }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              AI model running inference...
            </div>
          )}
        </Section>

        {result && (
          <Section title="Analysis Result">
            <div
              className="flex items-center gap-4 p-3 mb-3"
              style={{
                background: URGENCY_CONFIG[result.urgency_label]?.bg ?? "rgba(255,255,255,0.05)",
                border: `1px solid ${URGENCY_CONFIG[result.urgency_label]?.border ?? "rgba(255,255,255,0.1)"}`,
              }}
            >
              <div>
                <div
                  className="text-sm font-bold"
                  style={{ color: URGENCY_CONFIG[result.urgency_label]?.color ?? "#94a3b8", fontFamily: "'Exo 2', sans-serif" }}
                >
                  {URGENCY_CONFIG[result.urgency_label]?.label ?? result.urgency_label}
                </div>
                <div className="text-xs text-slate-500 mt-0.5" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                  Urgency Score: {(result.urgency_score * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
              <MetricBox label="Detected Class" value={TUMOR_LABELS[result.tumor_class]?.label ?? result.tumor_class} valueColor={TUMOR_LABELS[result.tumor_class]?.color} />
              <MetricBox label="Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} />
            </div>

            <div className="flex flex-col gap-2 mb-3">
              {Object.entries(result.class_probabilities ?? {}).map(([cls, prob]) => (
                <div key={cls}>
                  <div className="flex justify-between mb-1">
                    <span className="text-xs text-slate-400" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                      {TUMOR_LABELS[cls]?.label ?? cls}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1 w-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                    <div
                      className="h-full transition-all duration-700"
                      style={{ width: `${prob * 100}%`, background: TUMOR_LABELS[cls]?.color ?? "#00d4dc" }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={() => reportAPI.download(result.case_id, form.patient_name)}
              className="w-full py-2.5 text-xs font-bold tracking-widest text-black transition-opacity hover:opacity-90"
              style={{ background: "#00d4dc", fontFamily: "'Exo 2', sans-serif" }}
            >
              ↓ DOWNLOAD PDF REPORT
            </button>
          </Section>
        )}

        <button
          onClick={handleScan}
          disabled={isProcessing}
          className="w-full py-3 text-sm font-bold tracking-widest text-black transition-all mt-1"
          style={{
            background: isProcessing ? "rgba(0,212,220,0.45)" : "#00d4dc",
            fontFamily: "'Exo 2', sans-serif",
            letterSpacing: "0.12em",
            cursor: isProcessing ? "wait" : "pointer",
          }}
        >
          {step === "uploading" ? "↑ UPLOADING MRI..."
          : step === "analyzing" ? "◌ AI ANALYSIS RUNNING..."
          : "⊹ SCAN MRI"}
        </button>
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div
      className="mb-4 p-4"
      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
    >
      <h2
        className="text-xs font-semibold text-slate-300 mb-3"
        style={{ fontFamily: "'DM Sans', sans-serif" }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-slate-500" style={{ fontFamily: "'DM Sans', sans-serif" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

function MetricBox({ label, value, valueColor }) {
  return (
    <div
      className="px-3 py-2"
      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
    >
      <div className="text-xs text-slate-600 mb-1" style={{ fontFamily: "'DM Sans', sans-serif" }}>{label}</div>
      <div
        className="text-base font-bold"
        style={{ color: valueColor ?? "#e2e8f0", fontFamily: "'Exo 2', sans-serif" }}
      >
        {value}
      </div>
    </div>
  );
}