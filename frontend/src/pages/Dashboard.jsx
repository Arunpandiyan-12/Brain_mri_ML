import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import UploadPage    from "./UploadPage";
import QueuePage     from "./QueuePage";
import ResultsPage   from "./ResultsPage";
import ContactPage from "./ContactPage";
import PatientsInfoPage from "./PatientsInfo";

export default function Dashboard() {
  return (
    <div className="min-h-screen" style={{
      background: "radial-gradient(ellipse at 20% 10%, #0d2540 0%, #030712 55%, #000000 100%)",
      fontFamily: "'DM Sans', sans-serif",
    }}>
      {/* Background grid */}
      <div className="fixed inset-0 pointer-events-none opacity-5"
        style={{
          backgroundImage: "linear-gradient(rgba(0,229,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,255,0.3) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }} />

      {/* Glow top */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] pointer-events-none"
        style={{ background: "radial-gradient(ellipse, rgba(0,150,200,0.06) 0%, transparent 70%)" }} />

      <Navbar />

      <main className="relative z-10">
        <Routes>
          <Route path="/"          element={<Navigate to="/upload" replace />} />
          <Route path="/upload"    element={<UploadPage />} />
          <Route path="/queue"     element={<QueuePage />} />
          <Route path="/results"   element={<ResultsPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/patients" element={<PatientsInfoPage/>}/>
        </Routes>
      </main>
    </div>
  );
}
