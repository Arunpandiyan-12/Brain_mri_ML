import axios from "axios";

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1",
  timeout: 60000,
});

// Attach JWT token
API.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Handle 401
API.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

export default API;

// ── Auth ──────────────────────────────────────────────────────────────────────
export const authAPI = {
  login:    (d) => API.post("/auth/login", d),
  register: (d) => API.post("/auth/register", d),
  me:       ()  => API.get("/auth/me"),
};

// ── Cases ─────────────────────────────────────────────────────────────────────
export const casesAPI = {
  create: (d)  => API.post("/cases", d),
  list:   ()   => API.get("/cases"),
  get:    (id) => API.get(`/cases/${id}`),
  update: (id, d) => API.put(`/cases/${id}`, d),
  delete: (id) => API.delete(`/cases/${id}`),
};

// ── Scan ──────────────────────────────────────────────────────────────────────
export const scanAPI = {
  upload:  (caseId, formData) =>
    API.post(`/scan/upload/${caseId}`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  analyze: (caseId) => API.post(`/scan/analyze/${caseId}`),
  result:  (caseId) => API.get(`/scan/result/${caseId}`),
  status:  (caseId) => API.get(`/scan/status/${caseId}`),
  heatmapUrl: (caseId, type = "gradcam") =>
    `${API.defaults.baseURL}/scan/heatmap/${caseId}?type=${type}&token=${localStorage.getItem("token")}`,
};

// ── Queue ─────────────────────────────────────────────────────────────────────
export const queueAPI = {
  get:     ()  => API.get("/queue"),
  reorder: (d) => API.post("/queue/reorder", d),
  stats:   ()  => API.get("/queue/stats"),
};

// ── Report ────────────────────────────────────────────────────────────────────
export const reportAPI = {
  download: async (caseId, patientName = "report") => {
    const response = await API.get(`/report/${caseId}`, { responseType: "blob" });
    const url = window.URL.createObjectURL(new Blob([response.data], { type: "application/pdf" }));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `MRI_Report_${caseId}_${patientName.replace(/ /g, "_")}.pdf`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  },
};