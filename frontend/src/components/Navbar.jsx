import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import {
  Brain,
  Upload,
  Users,
  ListOrdered,
  Activity,
  Phone,
} from "lucide-react";

const NAV_ITEMS = [
  { label: "Upload Image", to: "/upload",   icon: <Upload size={13} strokeWidth={2} /> },
  { label: "Patient Info", to: "/patients", icon: <Users size={13} strokeWidth={2} /> },
  { label: "Queue Order",  to: "/queue",    icon: <ListOrdered size={13} strokeWidth={2} /> },
  { label: "Results",      to: "/results",  icon: <Activity size={13} strokeWidth={2} /> },
  { label: "Contact Us",   to: "/contact",  icon: <Phone size={13} strokeWidth={2} /> },
];

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <nav
      className="sticky top-0 z-50 flex items-center px-5 h-12 gap-0"
      style={{
        background: "rgba(10,12,24,0.98)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(20px)",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 mr-6 flex-shrink-0">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0"
          style={{ background: "rgba(0,212,220,0.15)", border: "1px solid rgba(0,212,220,0.35)" }}
        >
          <Brain size={15} color="#00d4dc" strokeWidth={1.5} />
        </div>
        <span
          className="text-xs font-bold text-white whitespace-nowrap tracking-wider uppercase"
          style={{ fontFamily: "'Exo 2', sans-serif" }}
        >
          Brain Tumor Detection
        </span>
      </div>

      {/* Nav links */}
      <div className="flex items-center gap-0.5 flex-1">
        {NAV_ITEMS.map(({ label, to, icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-all duration-150 whitespace-nowrap ${
                isActive ? "text-black" : "text-slate-400 hover:text-white"
              }`
            }
            style={({ isActive }) => ({
              background: isActive ? "#00d4dc" : "transparent",
              fontFamily: "'DM Sans', sans-serif",
            })}
          >
            {icon}
            {label}
          </NavLink>
        ))}
      </div>

      {/* User avatar */}
      <div className="ml-auto flex items-center gap-3 flex-shrink-0">
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold cursor-pointer hover:opacity-80 transition-opacity"
          style={{
            background: "rgba(0,212,220,0.15)",
            border: "1px solid rgba(0,212,220,0.3)",
            color: "#00d4dc",
            fontFamily: "'Exo 2', sans-serif",
          }}
          title={`${user?.username} · ${user?.role}`}
          onClick={handleLogout}
        >
          {user?.username?.[0]?.toUpperCase() ?? "A"}
        </div>
      </div>
    </nav>
  );
}