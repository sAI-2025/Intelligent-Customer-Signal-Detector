// static/js/data_upload.js — Data/Upload screen interactivity (Phase 7)

document.addEventListener("DOMContentLoaded", () => {

  // --- CSV dropzone + upload ---
  const csvDropzone = document.getElementById("csv-dropzone");
  const csvInput = document.getElementById("csv-input");
  const csvUploadBtn = document.getElementById("csv-upload-btn");
  let csvFile = null;

  setupDropzone(csvDropzone, csvInput, (file) => {
    csvFile = file;
    csvDropzone.querySelector("p").textContent = `Selected: ${file.name}`;
  });

  csvUploadBtn.addEventListener("click", async () => {
    if (!csvFile) { showToast("Please select a CSV file first."); return; }
    const formData = new FormData();
    formData.append("file", csvFile);
    csvUploadBtn.disabled = true;
    csvUploadBtn.textContent = "Uploading...";
    try {
      const res = await fetch("/api/upload/csv/", {
        method: "POST",
        headers: { "X-CSRFToken": window.CSRF_TOKEN },
        body: formData,
      });
      const data = await res.json();
      if (data.status === "ok") {
        showToast(`CSV processed: ${data.added} added, ${data.updated} updated, ${data.skipped} skipped.`);
        setTimeout(() => location.reload(), 1200);
      } else {
        showToast(`Upload failed: ${data.error || "unknown error"}`);
      }
    } catch (e) {
      showToast("Upload failed: network error.");
    } finally {
      csvUploadBtn.disabled = false;
      csvUploadBtn.textContent = "Upload CSV";
    }
  });

  // --- Transcript dropzone + upload ---
  const txtDropzone = document.getElementById("txt-dropzone");
  const txtInput = document.getElementById("txt-input");
  const txtUploadBtn = document.getElementById("txt-upload-btn");
  let txtFiles = [];

  setupDropzone(txtDropzone, txtInput, null, (files) => {
    txtFiles = Array.from(files);
    txtDropzone.querySelector("p").textContent = `Selected: ${txtFiles.length} file(s)`;
  });

  txtUploadBtn.addEventListener("click", async () => {
    if (!txtFiles.length) { showToast("Please select .txt file(s) first."); return; }
    const formData = new FormData();
    txtFiles.forEach((f) => formData.append("files", f));
    txtUploadBtn.disabled = true;
    txtUploadBtn.textContent = "Uploading...";
    try {
      const res = await fetch("/api/upload/transcripts/", {
        method: "POST",
        headers: { "X-CSRFToken": window.CSRF_TOKEN },
        body: formData,
      });
      const data = await res.json();
      if (data.status === "ok") {
        showToast(`Transcripts processed: ${data.linked} linked, ${data.pending} pending, ${data.failed} failed.`);
        setTimeout(() => location.reload(), 1200);
      } else {
        showToast(`Upload failed: ${data.error || "unknown error"}`);
      }
    } catch (e) {
      showToast("Upload failed: network error.");
    } finally {
      txtUploadBtn.disabled = false;
      txtUploadBtn.textContent = "Upload Transcripts";
    }
  });

  function setupDropzone(zone, input, onSingleFile, onMultiFile) {
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      const files = e.dataTransfer.files;
      if (onMultiFile) onMultiFile(files);
      else if (onSingleFile && files[0]) onSingleFile(files[0]);
    });
    input.addEventListener("change", () => {
      if (onMultiFile) onMultiFile(input.files);
      else if (onSingleFile && input.files[0]) onSingleFile(input.files[0]);
    });
  }

  // --- Manual customer entry ---
  const manualForm = document.getElementById("manual-form");
  if (manualForm) {
    manualForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(manualForm);
      try {
        const res = await fetch("/api/customer/manual/", {
          method: "POST",
          headers: { "X-CSRFToken": window.CSRF_TOKEN },
          body: formData,
        });
        const data = await res.json();
        if (data.status === "ok") {
          showToast(`Customer ${data.customer_id} ${data.created ? "added" : "updated"}.`);
          manualForm.reset();
        } else {
          showToast(`Failed: ${data.error || "unknown error"}`);
        }
      } catch (e) {
        showToast("Failed: network error.");
      }
    });
  }

  // --- Process button ---
  const processBtn = document.getElementById("process-btn");
  const progressBar = document.getElementById("process-progress");
  const progressFill = progressBar ? progressBar.querySelector(".progress-fill") : null;

  processBtn.addEventListener("click", async () => {
    processBtn.disabled = true;
    processBtn.textContent = "Processing...";
    progressBar.classList.remove("hidden");
    progressFill.style.width = "30%";
    try {
      const res = await fetch("/api/process/", {
        method: "POST",
        headers: { "X-CSRFToken": window.CSRF_TOKEN },
      });
      progressFill.style.width = "100%";
      const data = await res.json();
      if (data.status === "ok") {
        showToast(`${data.processed} customers processed · ${data.high_risk} flagged High Risk.`);
        setTimeout(() => { window.location.href = "/"; }, 1200);
      } else {
        showToast("Processing failed.");
      }
    } catch (e) {
      showToast("Processing failed: network error.");
    } finally {
      processBtn.disabled = false;
      processBtn.textContent = "Process Customers";
    }
  });

  // --- Danger zone ---
  document.querySelectorAll("[data-reset-target]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = btn.getAttribute("data-reset-target");
      const confirmed = prompt(`Type DELETE to confirm: this will remove ${target} data permanently.`);
      if (confirmed !== "DELETE") { showToast("Cancelled — confirmation text did not match."); return; }
      try {
        const formData = new FormData();
        formData.append("target", target);
        const res = await fetch("/api/data/reset/", {
          method: "POST",
          headers: { "X-CSRFToken": window.CSRF_TOKEN },
          body: formData,
        });
        const data = await res.json();
        if (data.status === "ok") {
          showToast(`Deleted: ${target}.`);
          setTimeout(() => location.reload(), 1000);
        } else {
          showToast("Reset failed.");
        }
      } catch (e) {
        showToast("Reset failed: network error.");
      }
    });
  });

});
