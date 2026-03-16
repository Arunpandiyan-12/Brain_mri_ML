import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import toast from "react-hot-toast";
import { FaBrain } from "react-icons/fa";
import { FiEye, FiEyeOff } from "react-icons/fi";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ identifier: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);

  const isEmail = (val) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.identifier.trim()) return toast.error("Username or email is required");
    if (!form.password)          return toast.error("Password is required");
    setLoading(true);
    try {
      const credential = isEmail(form.identifier)
        ? { email: form.identifier, password: form.password }
        : { username: form.identifier, password: form.password };
      await login(credential.username ?? credential.email, form.password);
      toast.success("Login successful");
      navigate("/");
    } catch {
      toast.error("Invalid credentials");
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
          <FaBrain size={28} color="#00d4dc" />
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
          Medical Imaging Analysis System
        </p>

        <form onSubmit={handleSubmit} className="w-full flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              Username or Email
            </label>
            <input
              type="text"
              value={form.identifier}
              onChange={(e) => setForm({ ...form, identifier: e.target.value })}
              placeholder="username or email@hospital.com"
              autoComplete="username"
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

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              Password
            </label>
            <div className="relative">
              <input
                type={showPass ? "text" : "password"}
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                placeholder="••••••••"
                autoComplete="current-password"
                className="w-full px-3 py-2.5 pr-10 text-sm text-white outline-none transition-colors"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  fontFamily: "'DM Sans', sans-serif",
                }}
                onFocus={e => (e.target.style.borderColor = "rgba(0,212,220,0.5)")}
                onBlur={e => (e.target.style.borderColor = "rgba(255,255,255,0.12)")}
              />
              <button
                type="button"
                tabIndex={-1}
                onClick={() => setShowPass(!showPass)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
              >
                {showPass ? <FiEyeOff size={15} /> : <FiEye size={15} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 text-sm font-semibold tracking-wider mt-1 transition-all"
            style={{
              background: loading ? "rgba(0,212,220,0.5)" : "#00d4dc",
              color: "#000",
              fontFamily: "'Exo 2', sans-serif",
              letterSpacing: "0.08em",
              cursor: loading ? "wait" : "pointer",
            }}
          >
            {loading ? "Authenticating..." : "Sign In"}
          </button>
        </form>

        <p className="text-xs text-slate-500 mt-6" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          New user?{" "}
          <Link to="/register" className="hover:underline" style={{ color: "#00d4dc" }}>
            Sign Up
          </Link>
        </p>
      </div>
    </div>
  );
}