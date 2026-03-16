import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { authAPI } from "../utils/api";
import toast from "react-hot-toast";

export default function RegisterPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await authAPI.register(form);
      toast.success("Account created");
      navigate("/login");
    } catch {
      toast.error("Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ background: "#0a0a0f" }}
    >
      <div
        className="w-full max-w-sm px-8 py-10 flex flex-col items-center"
        style={{
          background: "rgba(16,20,36,0.95)",
          border: "1px solid rgba(255,255,255,0.07)",
          boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
        }}
      >
        <div
          className="w-14 h-14 rounded-full flex items-center justify-center mb-4"
          style={{ background: "rgba(0,210,220,0.12)", border: "1.5px solid rgba(0,210,220,0.35)" }}
        >
          <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
            <path d="M16 4C13 4 10.5 5.5 9 8C7 8.5 5 10.5 5 13C3.5 14 2.5 15.5 2.5 17.5C2.5 20.5 5 23 8 23H16" stroke="#00d4dc" strokeWidth="1.8" strokeLinecap="round"/>
            <path d="M16 4C19 4 21.5 5.5 23 8C25 8.5 27 10.5 27 13C28.5 14 29.5 15.5 29.5 17.5C29.5 20.5 27 23 24 23H16" stroke="#00d4dc" strokeWidth="1.8" strokeLinecap="round"/>
            <path d="M16 4V23" stroke="#00d4dc" strokeWidth="1.5" strokeDasharray="3 2"/>
          </svg>
        </div>

        <h1
          className="text-lg font-bold text-white tracking-widest uppercase mb-1"
          style={{ fontFamily: "'Exo 2', sans-serif", letterSpacing: "0.18em" }}
        >
          Brain Tumor Detection
        </h1>
        <p
          className="text-xs mb-8 tracking-widest uppercase"
          style={{ color: "#00d4dc", fontFamily: "'DM Sans', sans-serif", letterSpacing: "0.15em" }}
        >
          Create Account
        </p>

        <form onSubmit={handleSubmit} className="w-full flex flex-col gap-4">
          {[
            { key: "username", label: "Username", type: "text", placeholder: "Enter username" },
            { key: "email",    label: "Email",    type: "email", placeholder: "your@email.com" },
            { key: "password", label: "Password", type: "password", placeholder: "••••••••" },
          ].map(({ key, label, type, placeholder }) => (
            <div key={key} className="flex flex-col gap-1">
              <label className="text-xs text-slate-400" style={{ fontFamily: "'DM Sans', sans-serif" }}>
                {label}
              </label>
              <input
                type={type}
                value={form[key]}
                onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                placeholder={placeholder}
                className="w-full px-3 py-2.5 text-sm text-white outline-none transition-colors"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  fontFamily: "'DM Sans', sans-serif",
                }}
                onFocus={e => (e.target.style.borderColor = "rgba(0,212,220,0.5)")}
                onBlur={e => (e.target.style.borderColor = "rgba(255,255,255,0.12)")}
              />
            </div>
          ))}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 text-sm font-semibold tracking-wider mt-1 transition-all"
            style={{
              background: loading ? "rgba(0,212,220,0.4)" : "#00d4dc",
              color: "#000",
              fontFamily: "'Exo 2', sans-serif",
              cursor: loading ? "wait" : "pointer",
            }}
          >
            {loading ? "Creating..." : "Register"}
          </button>
        </form>

        <p className="text-xs text-slate-500 mt-6" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          Already have an account?{" "}
          <Link to="/login" className="hover:underline" style={{ color: "#00d4dc" }}>
            Login
          </Link>
        </p>
      </div>
    </div>
  );
}