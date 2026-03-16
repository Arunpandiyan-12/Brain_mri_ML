import React from "react";
import {
  Github,
  Globe,
  MapPin,
  Brain,
  Code2,
  Server,
  Layout,
} from "lucide-react";

const TEAM = [
  {
    name: "Vidhya Shalini",
    role: "Backend Developer",
    focus: "API Design, ML Integration & Database",
    initials: "VS",
    color: "#06b6d4",
    bg: "#06b6d415",
    border: "#06b6d430",
    icon: <Server size={13} color="#06b6d4" />,
  },
  {
    name: "Padmini",
    role: "Frontend Developer",
    focus: "UI/UX, Dashboard & React Components",
    initials: "PA",
    color: "#34d399",
    bg: "#34d39915",
    border: "#34d39930",
    icon: <Layout size={13} color="#34d399" />,
  },
];

const STACK = [
  { label: "React",      color: "#61dafb" },
  { label: "FastAPI",    color: "#009688" },
  { label: "Python",     color: "#ffd43b" },
  { label: "TensorFlow", color: "#ff6f00" },
  { label: "Tailwind",   color: "#38bdf8" },
  { label: "PostgreSQL", color: "#74c0fc" },
];

export default function ContactPage() {
  return (
    <div
      style={{
        minHeight: "100%",
        background: "#05111e",
        padding: "32px 24px",
        fontFamily: "'Segoe UI', sans-serif",
      }}
    >
      {/* ── Project header ─────────────────────────────────────── */}
      <div
        style={{
          border: "1px solid #1a2a3a",
          background: "#060e1d",
          padding: "28px 32px",
          marginBottom: 24,
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* gradient top accent */}
        <div
          style={{
            position: "absolute",
            top: 0, left: 0, right: 0,
            height: 2,
            background: "linear-gradient(90deg, #06b6d4, #a78bfa, #34d399)",
          }}
        />

        <div style={{ display: "flex", alignItems: "flex-start", gap: 20 }}>
          {/* brain icon */}
          <div
            style={{
              width: 54, height: 54,
              background: "#06b6d415",
              border: "1px solid #06b6d430",
              display: "flex", alignItems: "center", justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Brain size={26} color="#06b6d4" strokeWidth={1.5} />
          </div>

          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 10, letterSpacing: "0.15em", color: "#475569", textTransform: "uppercase", marginBottom: 6 }}>
              Second Year Project · CSE Department
            </div>
            <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", margin: 0, marginBottom: 8 }}>
              Brain Tumor Detection System
            </h1>
            <p style={{ fontSize: 13, color: "#64748b", margin: 0, lineHeight: 1.7, maxWidth: 540 }}>
              An AI-powered diagnostic tool for detecting and classifying brain tumors from MRI scans
              using deep learning. Features Grad-CAM visualization, urgency triage, and automated
              clinical report generation.
            </p>

            {/* GitHub link */}
            <a
              href="https://github.com/vidhya-shalini/brain-scan-ai"
              target="_blank"
              rel="noreferrer"
              style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                marginTop: 14,
                fontSize: 11, color: "#38bdf8",
                textDecoration: "none",
                background: "#38bdf810",
                border: "1px solid #38bdf825",
                padding: "4px 12px",
              }}
            >
              <Github size={13} color="#38bdf8" />
              github.com/vidhya-shalini/brain-scan-ai
            </a>
          </div>

          {/* institution badge */}
          <div
            style={{
              background: "#081422",
              border: "1px solid #1a2a3a",
              padding: "14px 18px",
              textAlign: "center",
              flexShrink: 0,
              minWidth: 170,
            }}
          >
            <div
              style={{
                width: 40, height: 40,
                background: "#06b6d415",
                border: "1px solid #06b6d430",
                borderRadius: "50%",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 13, fontWeight: 800, color: "#06b6d4",
                margin: "0 auto 8px",
                letterSpacing: 1,
              }}
            >
              CIT
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 3 }}>
              Chennai Institute of Technology
            </div>
            <a
              href="https://citchennai.net"
              target="_blank"
              rel="noreferrer"
              style={{ fontSize: 11, color: "#38bdf8", textDecoration: "none" }}
            >
              citchennai.net
            </a>
            <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>B.E. Computer Science</div>
            <div style={{ fontSize: 11, color: "#475569" }}>2024 – 2028</div>
          </div>
        </div>
      </div>

      {/* ── Team cards ─────────────────────────────────────────── */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 10, letterSpacing: "0.15em", color: "#475569", textTransform: "uppercase", marginBottom: 14 }}>
          Project Team
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
          {TEAM.map((member) => (
            <div
              key={member.name}
              style={{
                background: "#060e1d",
                border: `1px solid ${member.border}`,
                padding: "20px",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* left accent */}
              <div
                style={{
                  position: "absolute",
                  top: 0, left: 0, bottom: 0,
                  width: 3,
                  background: member.color,
                  opacity: 0.7,
                }}
              />

              <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
                <div
                  style={{
                    width: 44, height: 44,
                    borderRadius: "50%",
                    background: member.bg,
                    border: `1px solid ${member.border}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, fontWeight: 700,
                    color: member.color,
                    flexShrink: 0,
                  }}
                >
                  {member.initials}
                </div>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: "#f1f5f9" }}>{member.name}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 3 }}>
                    {member.icon}
                    <span style={{ fontSize: 11, color: member.color }}>{member.role}</span>
                  </div>
                </div>
              </div>

              <div
                style={{
                  fontSize: 11, color: "#64748b",
                  background: "#081422",
                  padding: "8px 10px",
                  borderLeft: `2px solid ${member.border}`,
                  marginBottom: 12,
                  lineHeight: 1.5,
                }}
              >
                {member.focus}
              </div>

              <div style={{ display: "flex", gap: 8 }}>
                <div style={{ fontSize: 10, color: "#475569", background: "#081422", padding: "3px 8px", border: "1px solid #1a2a3a" }}>
                  CSE
                </div>
                <div style={{ fontSize: 10, color: "#475569", background: "#081422", padding: "3px 8px", border: "1px solid #1a2a3a" }}>
                  Second Year
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Tech stack + contact ────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Tech stack */}
        <div style={{ background: "#060e1d", border: "1px solid #1a2a3a", padding: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
            <Code2 size={13} color="#475569" />
            <span style={{ fontSize: 10, letterSpacing: "0.15em", color: "#475569", textTransform: "uppercase" }}>
              Tech Stack
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {STACK.map(({ label, color }) => (
              <div
                key={label}
                style={{
                  fontSize: 11, fontWeight: 600,
                  padding: "5px 12px",
                  background: color + "12",
                  border: `1px solid ${color}30`,
                  color,
                }}
              >
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Contact */}
        <div style={{ background: "#060e1d", border: "1px solid #1a2a3a", padding: 20 }}>
          <div style={{ fontSize: 10, letterSpacing: "0.15em", color: "#475569", textTransform: "uppercase", marginBottom: 14 }}>
            Project Contact
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 28, height: 28, background: "#081422", border: "1px solid #1a2a3a", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <MapPin size={13} color="#475569" />
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#475569", marginBottom: 1 }}>Institution</div>
                <div style={{ fontSize: 12, color: "#94a3b8" }}>Chennai Institute of Technology, Chennai</div>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 28, height: 28, background: "#081422", border: "1px solid #1a2a3a", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <Github size={13} color="#475569" />
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#475569", marginBottom: 1 }}>GitHub</div>
                <a
                  href="https://github.com/vidhya-shalini/brain-scan-ai"
                  target="_blank"
                  rel="noreferrer"
                  style={{ fontSize: 12, color: "#38bdf8", textDecoration: "none" }}
                >
                  github.com/vidhya-shalini/brain-scan-ai
                </a>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 28, height: 28, background: "#081422", border: "1px solid #1a2a3a", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <Globe size={13} color="#475569" />
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#475569", marginBottom: 1 }}>Website</div>
                <a
                  href="https://citchennai.net"
                  target="_blank"
                  rel="noreferrer"
                  style={{ fontSize: 12, color: "#38bdf8", textDecoration: "none" }}
                >
                  citchennai.net
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}