// Placeholder frontend logic. Points at the local FastAPI backend by default.
const API_BASE = "http://localhost:8000";

async function checkBackend() {
  const statusEl = document.getElementById("status");
  statusEl.textContent = "checking...";
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    statusEl.textContent = data.status === "healthy" ? "connected ✅" : "unexpected response";
  } catch (err) {
    statusEl.textContent = "not reachable ❌ (start the backend: uvicorn app.main:app --reload)";
  }
}

document.getElementById("check-btn").addEventListener("click", checkBackend);
checkBackend();
