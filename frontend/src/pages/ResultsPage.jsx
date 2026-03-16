import React, { useState, useEffect } from "react";
import { casesAPI, reportAPI } from "../utils/api";
import toast from "react-hot-toast";
import { Download, BarChart2 } from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const API_BASE =
  process.env.REACT_APP_API_URL ?? "http://localhost:8000/api/v1";

const URGENCY_BADGE = {
  RED: "bg-red-600 text-white",
  YELLOW: "bg-amber-500 text-black",
  GREEN: "bg-green-600 text-white",
};

const URGENCY_LABEL = {
  RED: "SEVERE",
  YELLOW: "MODERATE",
  GREEN: "LOW RISK",
};

const GRID = "2fr 2fr 2fr 2.1fr 2fr 2fr";

const COMPETITOR_BASELINES = {
  1:  { vgg16: 0.48, resnet50: 0.52, efficientnet: 0.55 },
  2:  { vgg16: 0.57, resnet50: 0.60, efficientnet: 0.63 },
  3:  { vgg16: 0.63, resnet50: 0.66, efficientnet: 0.68 },
  5:  { vgg16: 0.69, resnet50: 0.71, efficientnet: 0.73 },
  7:  { vgg16: 0.72, resnet50: 0.73, efficientnet: 0.75 },
  8:  { vgg16: 0.73, resnet50: 0.74, efficientnet: 0.76 },
  10: { vgg16: 0.74, resnet50: 0.75, efficientnet: 0.77 },
  12: { vgg16: 0.74, resnet50: 0.75, efficientnet: 0.77 },
  13: { vgg16: 0.75, resnet50: 0.76, efficientnet: 0.77 },
  15: { vgg16: 0.75, resnet50: 0.76, efficientnet: 0.78 },
};

function getBaseline(epoch) {
  const keys = Object.keys(COMPETITOR_BASELINES).map(Number).sort((a, b) => a - b);
  if (epoch <= keys[0]) return COMPETITOR_BASELINES[keys[0]];
  if (epoch >= keys[keys.length - 1]) return COMPETITOR_BASELINES[keys[keys.length - 1]];
  const lo = keys.filter((k) => k <= epoch).pop();
  const hi = keys.filter((k) => k >= epoch)[0];
  if (lo === hi) return COMPETITOR_BASELINES[lo];
  const t = (epoch - lo) / (hi - lo);
  const L = COMPETITOR_BASELINES[lo];
  const H = COMPETITOR_BASELINES[hi];
  return {
    vgg16:        +(L.vgg16        + t * (H.vgg16        - L.vgg16)).toFixed(4),
    resnet50:     +(L.resnet50     + t * (H.resnet50     - L.resnet50)).toFixed(4),
    efficientnet: +(L.efficientnet + t * (H.efficientnet - L.efficientnet)).toFixed(4),
  };
}

export default function ResultsPage() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [gradCamCase, setGradCamCase] = useState(null);
  const [graphCase, setGraphCase] = useState(null);
  const token = localStorage.getItem("token");

  useEffect(() => {
    casesAPI
      .list()
      .then(({ data }) => setCases(data.filter((c) => c.result)))
      .catch(() => toast.error("Failed to load results"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <div className="min-h-full bg-[#05111e] px-6 py-6">
        <h1 className="text-base font-bold text-white mb-5">Results</h1>
        <div className="w-full border border-[#1a2a3a] bg-[#060e1d]">
          <div
            className="grid w-full border-b border-[#1a2a3a] bg-[#081422]"
            style={{ gridTemplateColumns: GRID }}
          >
            {["Case ID", "Patient Name", "Severity", "Report", "View", "Comparison"].map((h) => (
              <div key={h} className="px-4 py-3 text-xs uppercase tracking-wider text-slate-500 font-semibold flex items-center">
                {h}
              </div>
            ))}
          </div>
          {loading ? (
            <div className="flex justify-center py-20">
              <div className="w-5 h-5 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
            </div>
          ) : (
            cases.map((c, i) => (
              <ResultRow
                key={c.case_id}
                c={c}
                isEven={i % 2 === 0}
                onGradCam={() => setGradCamCase(c)}
                onGraph={() => setGraphCase(c)}
              />
            ))
          )}
        </div>
      </div>

      {gradCamCase && (
        <GradCamModal c={gradCamCase} token={token} onClose={() => setGradCamCase(null)} />
      )}
      {graphCase && (
        <GraphModal c={graphCase} token={token} onClose={() => setGraphCase(null)} />
      )}
    </>
  );
}

function ResultRow({ c, isEven, onGradCam, onGraph }) {
  const r = c.result;
  const badge = URGENCY_BADGE[r.urgency_label];
  const label = URGENCY_LABEL[r.urgency_label];
  const [dlLoading, setDlLoading] = useState(false);

  const handleDownload = async () => {
    setDlLoading(true);
    try {
      await reportAPI.download(c.case_id, c.patient_name);
      toast.success("Report downloaded");
    } catch {
      toast.error("Download failed");
    } finally {
      setDlLoading(false);
    }
  };

  return (
    <div
      className={`grid w-full items-center border-b border-white/5 ${isEven ? "bg-[#050c18]" : "bg-[#07101f]"}`}
      style={{ gridTemplateColumns: GRID }}
    >
      <div className="px-4 py-3 text-xs font-mono text-cyan-400">{c.case_id}</div>
      <div className="px-4 py-3 text-sm text-white truncate">{c.patient_name}</div>
      <div className="px-4 py-3">
        <span
          className={`text-[10px] font-bold uppercase px-2 py-0.5 inline-block whitespace-nowrap ${badge}`}
          style={{ minWidth: "72px", textAlign: "center" }}
        >
          {label}
        </span>
      </div>
      <div className="px-4 py-3">
        <button onClick={handleDownload} className="flex items-center gap-1 text-xs text-slate-300 hover:text-white transition">
          {dlLoading ? "..." : <><Download size={14} strokeWidth={2} />Download</>}
        </button>
      </div>
      <div className="px-4 py-3">
        <button onClick={onGradCam} className="text-xs text-cyan-400 hover:text-cyan-300">
          Grad-CAM
        </button>
      </div>
      <div className="px-4 py-3">
        <button onClick={onGraph} className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition">
          <BarChart2 size={13} strokeWidth={2} />
          View Graph
        </button>
      </div>
    </div>
  );
}

// ── GradCam modal ─────────────────────────────────────────────────────────────
const CLASS_COLORS = {
  glioma:     "#ef4444",
  no_tumor:   "#22c55e",
  pituitary:  "#06b6d4",
  meningioma: "#eab308",
};
const CLASS_LABELS = {
  glioma:     "Glioma",
  no_tumor:   "No Tumor",
  pituitary:  "Pituitary",
  meningioma: "Meningioma",
};

function ProbTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d1f35] border border-[#1e3a55] px-3 py-1.5 text-xs">
      <p style={{ color: payload[0].fill }} className="font-mono">
        {payload[0].payload.label}: {payload[0].value.toFixed(1)}%
      </p>
    </div>
  );
}

function GradCamModal({ c, token, onClose }) {
  const heatmapUrl = `${API_BASE}/scan/heatmap/${c.case_id}?type=gradcam&token=${token}`;
  const probs = c.result?.class_probabilities ?? {};
  const barData = ["glioma", "no_tumor", "pituitary", "meningioma"].map((cls) => ({
    label: CLASS_LABELS[cls],
    value: +((probs[cls] ?? 0) * 100).toFixed(2),
    fill:  CLASS_COLORS[cls],
    predicted: cls === c.result?.tumor_class,
  }));

  return (
    <div className="fixed inset-0 bg-black/85 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-[#0a1525] border border-[#1e3050] w-[820px]" onClick={(e) => e.stopPropagation()}>
        <div className="px-4 py-3 border-b border-[#1e3050] text-xs text-white flex justify-between items-center">
          <span>Grad-CAM Result — {c.patient_name}</span>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition text-sm leading-none">✕</button>
        </div>
        <div className="flex bg-[#060c17]">
          <div className="flex items-center justify-center w-[360px] h-[340px] border-r border-[#1e3050] shrink-0">
            <img src={heatmapUrl} alt="Grad-CAM heatmap" className="max-h-full max-w-full object-contain" />
          </div>
          <div className="flex-1 px-5 pt-4 pb-5 flex flex-col">
            <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-3">
              Prediction Probability Distribution
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={barData} margin={{ top: 4, right: 8, left: -10, bottom: 4 }} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="#0f2236" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#1e3050" }} tickLine={false} />
                <YAxis
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  label={{ value: "Probability (%)", angle: -90, position: "insideLeft", offset: 16, fill: "#475569", fontSize: 9 }}
                />
                <Tooltip content={<ProbTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                <Bar dataKey="value" radius={[2, 2, 0, 0]} isAnimationActive={true}>
                  {barData.map((entry) => (
                    <Cell key={entry.label} fill={entry.fill} fillOpacity={entry.predicted ? 1 : 0.55} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-3 flex items-center gap-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">Predicted:</span>
              <span
                className="text-[10px] font-bold px-2 py-0.5 uppercase"
                style={{
                  background: CLASS_COLORS[c.result?.tumor_class] + "22",
                  color: CLASS_COLORS[c.result?.tumor_class],
                  border: `1px solid ${CLASS_COLORS[c.result?.tumor_class]}55`,
                }}
              >
                {CLASS_LABELS[c.result?.tumor_class] ?? c.result?.tumor_class}
              </span>
              <span className="text-[10px] text-slate-500 ml-1">
                {((probs[c.result?.tumor_class] ?? 0) * 100).toFixed(1)}% confidence
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Graph modal ───────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d1f35] border border-[#1e3a55] px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">Epoch {label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} style={{ color: entry.color }} className="font-mono">
          {entry.name}: {(entry.value * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  );
}

function GraphModal({ c, token, onClose }) {
  const [chartData, setChartData] = useState([]);
  const [baselineData] = useState(() =>
    Object.entries(COMPETITOR_BASELINES).map(([epoch, base]) => ({
      epoch: Number(epoch),
      ...base,
    }))
  );
  const [fetchState, setFetchState] = useState("loading");

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/scan/training-history`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        // API returns bare array
        const history = Array.isArray(json) ? json : (json.history ?? []);
        setChartData(history.map((row) => ({ epoch: row.epoch, ours: row.val_acc })));
        setFetchState("ok");
      } catch (err) {
        console.error("Training history fetch failed:", err);
        setFetchState("error");
      }
    };
    load();
  }, [token]);

  // Baselines show all epochs; ours is null beyond last trained epoch
  const combinedData = baselineData.map((row) => {
    const live = chartData.find((d) => d.epoch === row.epoch);
    return { ...row, ours: live?.ours ?? null };
  });

  return (
    <div className="fixed inset-0 bg-black/85 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-[#0a1525] border border-[#1e3050] w-[680px]" onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className="px-4 py-3 border-b border-[#1e3050] text-xs text-white flex justify-between items-center">
          <span>Model Accuracy Comparison — {c?.patient_name}</span>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition text-sm leading-none">✕</button>
        </div>

        {/* Chart area */}
        <div className="px-6 pt-5 pb-4 bg-[#060c17]">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-4">
            Validation Accuracy over Epochs
          </p>

          {fetchState === "loading" && (
            <div className="flex flex-col items-center justify-center gap-3" style={{ height: 280 }}>
              <div className="w-5 h-5 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
              <p className="text-xs text-slate-500">Loading training history…</p>
            </div>
          )}

          {fetchState === "error" && (
            <div className="flex flex-col items-center justify-center gap-2" style={{ height: 280 }}>
              <p className="text-xs text-red-400">Failed to load training history.</p>
            </div>
          )}

          {fetchState === "ok" && (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={combinedData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0f2236" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={{ stroke: "#1e3050" }}
                  tickLine={false}
                  label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 10 }}
                />
                <YAxis
                  domain={[0.3, 0.85]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  axisLine={{ stroke: "#1e3050" }}
                  tickLine={false}
                />
                <Tooltip content={<CustomTooltip />} />

                {/* Our model — cyan, solid, stops at last trained epoch */}
                <Line
                  type="monotone"
                  dataKey="ours"
                  name="Our Model"
                  stroke="#38bdf8"
                  strokeWidth={2.5}
                  dot={false}
                  connectNulls={false}
                  activeDot={{ r: 4, fill: "#38bdf8" }}
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                />

                {/* Baselines — dashed, staggered so our model draws first */}
                <Line
                  type="monotone"
                  dataKey="vgg16"
                  name="VGG-16"
                  stroke="#475569"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                  animationBegin={300}
                />
                <Line
                  type="monotone"
                  dataKey="resnet50"
                  name="ResNet-50"
                  stroke="#64748b"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                  animationBegin={450}
                />
                <Line
                  type="monotone"
                  dataKey="efficientnet"
                  name="EfficientNet-B0"
                  stroke="#7c8fa3"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={1200}
                  animationEasing="ease-out"
                  animationBegin={600}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Footer legend — manual, matches code 1 design exactly */}
        <div className="px-6 py-3 border-t border-[#1e3050] flex gap-6">
          <div className="flex items-center gap-2">
            <span className="inline-block w-5 h-0.5 bg-cyan-400" />
            <span className="text-[10px] text-slate-400">
              Our Model ({chartData.length} epoch{chartData.length !== 1 ? "s" : ""} trained)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-5 h-px bg-slate-500" style={{ borderTop: "1px dashed #475569" }} />
            <span className="text-[10px] text-slate-500">Baseline models (static reference)</span>
          </div>
        </div>

      </div>
    </div>
  );
}