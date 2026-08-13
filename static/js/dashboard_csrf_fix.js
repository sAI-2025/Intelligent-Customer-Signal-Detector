// static/js/dashboard.js — CSRF fixed using meta tag

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<span>${message}</span><span class="toast-close">&times;</span>`;
  toast.querySelector(".toast-close").onclick = () => toast.remove();
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function getCsrfToken() {
  const meta = document.querySelector('meta[name="csrf-token"]');
  return meta ? meta.getAttribute("content") : "";
}

window.CSRF_TOKEN = getCsrfToken();
