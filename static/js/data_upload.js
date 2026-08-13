// static/js/data_upload.js — Data/Upload screen interactivity (Phase 7)

document.addEventListener("DOMContentLoaded", () => {
  const pageData = document.getElementById("data-upload-page").dataset;

  // --- CSV & Transcript dropzones ---
  const csvDropzone = document.getElementById("csv-dropzone");
  const csvInput = document.getElementById("csv-input");
  let csvFile = null;

  const txtDropzone = document.getElementById("txt-dropzone");
  const txtInput = document.getElementById("txt-input");
  let txtFiles = [];

  setupDropzone(csvDropzone, csvInput, (file) => {
    csvFile = file;
    csvDropzone.querySelector("p").textContent = `Selected: ${file.name}`;
  });

  setupDropzone(txtDropzone, txtInput, null, (files) => {
    txtFiles = Array.from(files);
    txtDropzone.querySelector("p").textContent =
      `Selected: ${txtFiles.length} file(s)`;
  });

  // single Upload button handles both CSV and transcripts
  const uploadBtn = document.getElementById("upload-btn");
  if (uploadBtn) {
    uploadBtn.addEventListener("click", async () => {
      if (!csvFile && txtFiles.length === 0) {
        showToast(
          "Please select a CSV or one or more .txt files before uploading.",
        );
        return;
      }
      const formData = new FormData();
      if (csvFile) formData.append("file", csvFile);
      txtFiles.forEach((f) => formData.append("files", f));
      const auto = document.getElementById("auto-process");
      if (auto && auto.checked) formData.append("auto_process", "1");

      uploadBtn.disabled = true;
      uploadBtn.textContent = "Uploading...";
      try {
        const res = await fetch(pageData.uploadUrl, {
          method: "POST",
          headers: { "X-CSRFToken": window.CSRF_TOKEN },
          body: formData,
        });
        const data = await res.json();
        if (data.status === "ok") {
          let msg = "Upload complete.";
          if (data.csv) {
            msg = `CSV: ${data.csv.added || 0} added, ${data.csv.updated || 0} updated, ${data.csv.skipped || 0} skipped.`;
          }
          if (typeof data.linked !== "undefined") {
            msg += ` Transcripts: ${data.linked} linked, ${data.pending} pending, ${data.failed} failed.`;
          }
          if (data.pipeline) {
            msg += ` Processing: ${data.pipeline.processed} processed · ${data.pipeline.high_risk} high risk.`;
          }
          showToast(msg);
          setTimeout(() => location.reload(), 1200);
        } else {
          showToast(`Upload failed: ${data.error || "unknown error"}`);
        }
      } catch (e) {
        showToast("Upload failed: network error.");
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload";
      }
    });
  }

  function setupDropzone(zone, input, onSingleFile, onMultiFile) {
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
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
        const res = await fetch(pageData.manualUrl, {
          method: "POST",
          headers: { "X-CSRFToken": window.CSRF_TOKEN },
          body: formData,
        });
        const data = await res.json();
        if (data.status === "ok") {
          showToast(
            `Customer ${data.customer_id} ${data.created ? "added" : "updated"}.`,
          );
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
  const progressFill = progressBar
    ? progressBar.querySelector(".progress-fill")
    : null;

  processBtn.addEventListener("click", async () => {
    processBtn.disabled = true;
    processBtn.textContent = "Processing...";
    progressBar.classList.remove("hidden");
    progressFill.style.width = "30%";
    try {
      const res = await fetch(pageData.processUrl, {
        method: "POST",
        headers: { "X-CSRFToken": window.CSRF_TOKEN },
      });
      progressFill.style.width = "100%";
      const data = await res.json();
      if (data.status === "ok") {
        showToast(
          `${data.processed} customers processed · ${data.high_risk} flagged High Risk.`,
        );
        setTimeout(() => {
          window.location.href = "/";
        }, 1200);
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
      const confirmed = prompt(
        `Type DELETE to confirm: this will remove ${target} data permanently.`,
      );
      if (confirmed !== "DELETE") {
        showToast("Cancelled — confirmation text did not match.");
        return;
      }
      try {
        const formData = new FormData();
        formData.append("target", target);
        const res = await fetch(pageData.resetUrl, {
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
