/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#030712",
          900: "#060e1e",
          800: "#0a1628",
          700: "#0d1f38",
          600: "#112848",
          500: "#163358",
        },
        cyan: {
          400: "#22d3ee",
          500: "#06b6d4",
          glow: "#00e5ff",
        },
        critical: "#ff4444",
        warning:  "#ffaa00",
        safe:     "#00e676",
      },
      fontFamily: {
        display: ["'Exo 2'", "sans-serif"],
        body:    ["'DM Sans'", "sans-serif"],
        mono:    ["'JetBrains Mono'", "monospace"],
      },
      backgroundImage: {
        "radial-navy": "radial-gradient(ellipse at top left, #0d1f38 0%, #030712 70%)",
        "radial-glow":  "radial-gradient(circle at 50% 0%, rgba(0,229,255,0.08) 0%, transparent 60%)",
        "card-glass":   "linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)",
      },
      boxShadow: {
        "glow-cyan":  "0 0 20px rgba(0,229,255,0.35), 0 0 40px rgba(0,229,255,0.15)",
        "glow-red":   "0 0 20px rgba(255,68,68,0.4)",
        "glow-green": "0 0 20px rgba(0,230,118,0.3)",
        "card":       "0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05)",
      },
      animation: {
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
        "slide-up":   "slideUp 0.4s ease-out",
        "fade-in":    "fadeIn 0.3s ease-out",
        "spin-slow":  "spin 3s linear infinite",
        "scan-line":  "scanLine 2s linear infinite",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 10px rgba(0,229,255,0.3)" },
          "50%":       { boxShadow: "0 0 30px rgba(0,229,255,0.7), 0 0 60px rgba(0,229,255,0.3)" },
        },
        slideUp: {
          from: { transform: "translateY(12px)", opacity: 0 },
          to:   { transform: "translateY(0)",    opacity: 1 },
        },
        fadeIn: {
          from: { opacity: 0 },
          to:   { opacity: 1 },
        },
        scanLine: {
          "0%":   { top: "0%" },
          "100%": { top: "100%" },
        },
      },
    },
  },
  plugins: [],
};
