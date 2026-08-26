const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  bootstrap: null,
  jobs: [],
  activeJobId: null,
  eventSource: null,
  duration: 5,
  imageAsset: null,
  dialogTrigger: null,
  resourceTimer: null,
  elapsedTimer: null,
  promptTransform: null,
  dialogJobId: null,
  deleteJobId: null,
  deleteDialogTrigger: null,
};

const viewMeta = {
  generate: ["CREATE", "新しい映像をつくる"],
  queue: ["JOBS", "生成キュー"],
  library: ["LIBRARY", "生成ライブラリ"],
  settings: ["SYSTEM", "エンジン設定"],
};

const statusLabels = {
  queued: "待機中", running: "生成中", stopping: "途中保存中", cancelling: "停止中",
  succeeded: "完了", partial: "途中保存", failed: "失敗", cancelled: "中止", interrupted: "中断",
};

const THEME_KEY = "hayate-studio-theme";

function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) || "dark";
  } catch {
    return "dark";
  }
}

function applyTheme(theme, persist = true) {
  const selected = theme === "clear" ? "clear" : "dark";
  document.documentElement.dataset.theme = selected;
  const button = $("#themeButton");
  if (button) {
    const clear = selected === "clear";
    button.setAttribute("aria-pressed", String(clear));
    button.title = clear ? "Darkモードに切り替え" : "Clearモードに切り替え";
    button.setAttribute("aria-label", button.title);
    $("#themeLabel").textContent = clear ? "DARK" : "CLEAR";
  }
  document.querySelector('meta[name="theme-color"]')?.setAttribute(
    "content", selected === "clear" ? "#eef5fb" : "#070b12"
  );
  if (persist) {
    try { localStorage.setItem(THEME_KEY, selected); } catch { /* optional */ }
  }
}

async function api(path, options = {}) {
  const request = { ...options, headers: { ...(options.headers || {}) } };
  if (request.method && request.method !== "GET") request.headers["X-HAYATE-UI"] = "1";
  if (request.body && !(request.body instanceof FormData) && typeof request.body !== "string") {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(request.body);
  }
  const response = await fetch(path, request);
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail || payload);
    throw new Error(message || `HTTP ${response.status}`);
  }
  return payload;
}

function toast(message, kind = "success") {
  const element = document.createElement("div");
  element.className = `toast ${kind}`;
  element.textContent = message;
  $("#toastStack").append(element);
  setTimeout(() => element.remove(), 4500);
}

function bytes(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = Number(value);
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1; }
  return `${amount.toFixed(index >= 3 ? 1 : 0)} ${units[index]}`;
}

function clock(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "計算中";
  const total = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}` : `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}

function compactDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("ja-JP", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function framesFor(seconds) {
  return 17 * Math.max(1, Math.round((seconds * 24 - 5) / 17)) + 5;
}

function navigate(view) {
  if (!viewMeta[view]) return;
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
  $("#viewEyebrow").textContent = viewMeta[view][0];
  $("#viewTitle").textContent = viewMeta[view][1];
  history.replaceState(null, "", `#${view}`);
  if (view === "queue" || view === "library") refreshJobs();
}

function selectedProfile() {
  return $('input[name="profile"]:checked')?.value || "fast_sage";
}

function applyProfile(profile) {
  const preset = state.bootstrap?.profiles?.[profile];
  if (preset) {
    $("#steps").value = preset.steps;
    $("#attention").value = preset.attention_backend;
    $("#easycache").checked = preset.easycache;
    $("#ecThreshold").value = preset.easycache_threshold;
    $("#ecStart").value = preset.easycache_start;
    $("#ecEnd").value = preset.easycache_end;
    $("#ecSkips").value = preset.easycache_max_consecutive_skips;
    $("#blocksSwap").value = preset.blocks_to_swap;
    $("#chunkRows").value = preset.activation_chunk_rows;
    $("#vaeTile").value = preset.vae_tile_size;
    updateProfileSummary();
  } else {
    updateProfileSummary();
  }
}

function updateProfileSummary() {
  const parts = [`${$("#steps").value} points`, $("#attention").value === "sageattn" ? "SageAttention" : "SDPA"];
  if ($("#easycache").checked) parts.push("EasyCache");
  $("#profileSummary").textContent = parts.join(" · ");
}

function updateDuration() {
  const count = framesFor(state.duration);
  const exact = count / 24;
  $("#frameHint").textContent = `${count} frames`;
  $("#summaryFrames").textContent = `${count} frames`;
  $("#summaryDuration").textContent = `${exact.toFixed(1)}秒`;
}

function resolution() {
  const preset = $("#resolutionPreset").value;
  if (preset === "custom") return [Number($("#width").value), Number($("#height").value)];
  return preset.split("x").map(Number);
}

function updateResolution() {
  const custom = $("#resolutionPreset").value === "custom";
  $("#customResolution").hidden = !custom;
  const [width, height] = resolution();
  $("#resolutionHint").textContent = `${width} × ${height}`;
}

async function uploadImage(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const dropzone = $("#dropzone");
  dropzone.classList.add("dragging");
  try {
    const asset = await api("/api/assets", { method: "POST", body: form });
    state.imageAsset = asset;
    $("#imagePreview").src = asset.url;
    $("#imagePreview").hidden = false;
    $("#removeImage").hidden = false;
    $("#task").value = "auto";
    toast("開始画像を読み込みました");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    dropzone.classList.remove("dragging");
  }
}

function removeImage() {
  state.imageAsset = null;
  $("#imageFile").value = "";
  $("#imagePreview").removeAttribute("src");
  $("#imagePreview").hidden = true;
  $("#removeImage").hidden = true;
}

function generationPayload() {
  const [width, height] = resolution();
  return {
    prompt: $("#prompt").value.trim(),
    original_prompt: state.promptTransform?.original || null,
    prompt_transform_applied: Boolean(state.promptTransform),
    prompt_transform_template_version: state.promptTransform?.version || null,
    profile: selectedProfile(),
    task: $("#task").value,
    width, height,
    duration_seconds: state.duration,
    seed: Number($("#seed").value),
    filename: "hayate",
    image_asset_id: state.imageAsset?.id || null,
    last_image_asset_id: null,
    reference_asset_ids: [],
    use_prompt_cache: $("#promptCache").checked,
    steps: Number($("#steps").value),
    attention_backend: $("#attention").value,
    easycache: $("#easycache").checked,
    easycache_threshold: Number($("#ecThreshold").value),
    easycache_start: Number($("#ecStart").value),
    easycache_end: Number($("#ecEnd").value),
    easycache_max_consecutive_skips: Number($("#ecSkips").value),
    blocks_to_swap: Number($("#blocksSwap").value),
    activation_chunk_rows: Number($("#chunkRows").value),
    vae_tile_size: Number($("#vaeTile").value),
  };
}

function structuredPromptDraft() {
  const visual = $("#promptVisual").value.trim();
  const sound = $("#promptSound").value.trim();
  const music = $("#promptMusic").value.trim();
  const instruction = state.imageAsset
    ? "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    : "";
  const draft = `${instruction}integrated_multimodal_description: [Shot 1] ${visual}\n\noverall_soundscape: ${sound}\n\nnon_diegetic_music: ${music}`;
  $("#promptDraft").value = draft;
  return { draft, visual, sound, music };
}

function openPromptAssist() {
  const original = state.promptTransform?.original || $("#prompt").value.trim();
  if (!original) return toast("先に映像の内容を入力してください", "error");
  $("#promptVisual").value = original;
  $("#promptSound").value = "";
  $("#promptMusic").value = "";
  structuredPromptDraft();
  state.dialogTrigger = document.activeElement;
  $("#promptDialog").showModal();
}

function applyStructuredPrompt() {
  const { draft, visual, sound, music } = structuredPromptDraft();
  if (!visual || !sound || !music) {
    return toast(
      "映像・環境音・背景音楽をすべて入力してください。不要な項目は N/A と入力できます",
      "error"
    );
  }
  const original = state.promptTransform?.original || $("#prompt").value.trim();
  state.promptTransform = { original, version: "h3-base-fields-v1" };
  $("#prompt").value = draft;
  $("#prompt").dispatchEvent(new Event("input"));
  $("#promptDialog").close();
  toast("確認したH3向けプロンプト案を適用しました");
}

async function submitGeneration(event) {
  event?.preventDefault();
  const button = $("#generateButton");
  const payload = generationPayload();
  if (!payload.prompt) return toast("プロンプトを入力してください", "error");
  button.disabled = true;
  button.querySelector("span:nth-of-type(2)").textContent = "プリフライト中…";
  try {
    const job = await api("/api/jobs", { method: "POST", body: payload });
    upsertJob(job);
    state.activeJobId = job.id;
    renderLive(job);
    connectEvents(job.id);
    renderJobs();
    toast("生成キューへ追加しました");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.querySelector("span:nth-of-type(2)").textContent = "生成を開始";
  }
}

function upsertJob(job) {
  const index = state.jobs.findIndex((item) => item.id === job.id);
  if (index >= 0) state.jobs[index] = job;
  else state.jobs.unshift(job);
  state.jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
}

function connectEvents(jobId) {
  state.eventSource?.close();
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;
  source.onmessage = (event) => {
    const job = JSON.parse(event.data);
    upsertJob(job);
    if (state.activeJobId === job.id) renderLive(job);
    renderJobs();
    if (["succeeded", "partial", "failed", "cancelled", "interrupted"].includes(job.status)) {
      source.close();
      if (["succeeded", "partial"].includes(job.status)) toast(job.status === "partial" ? "途中状態を保存しました" : "動画が完成しました");
      else if (job.status === "failed") toast(job.error || "生成に失敗しました", "error");
      refreshJobs();
    }
  };
  source.onerror = () => {
    source.close();
    setTimeout(() => refreshJobs(), 1200);
  };
}

function activeElapsed(job) {
  if (!job.started_at) return 0;
  const end = job.completed_at ? new Date(job.completed_at) : new Date();
  return Math.max(0, (end - new Date(job.started_at)) / 1000);
}

function renderLive(job) {
  const idle = !job;
  $("#idleState").hidden = !idle;
  $("#progressState").hidden = idle;
  $("#liveCard").classList.toggle("running", !idle && ["queued", "running", "stopping", "cancelling"].includes(job.status));
  if (idle) {
    $("#liveStatus").className = "status-badge idle";
    $("#liveStatus").textContent = "待機中";
    return;
  }
  const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
  $("#liveStatus").className = `status-badge ${job.status}`;
  $("#liveStatus").textContent = statusLabels[job.status] || job.status;
  $("#liveStage").textContent = job.stage || "準備中";
  $("#liveDetail").textContent = job.detail || "—";
  $("#livePercent").textContent = `${Math.round(progress)}%`;
  $("#liveProgressBar").value = progress;
  $("#liveProgressTrack").setAttribute("aria-valuenow", String(Math.round(progress)));
  $("#elapsedTime").textContent = clock(activeElapsed(job));
  $("#etaTime").textContent = job.eta_seconds === 0 ? "完了" : clock(job.eta_seconds);
  $("#liveProfile").textContent = String(job.request?.profile || "history")
    .replace("fast_sage_detail", "高速・画質優先")
    .replace("fast_sage", "最速")
    .replace("fast", "Fast SDPA")
    .replace("quality", "Quality");
  $$("#stageList span").forEach((span) => span.classList.toggle("done", progress >= Number(span.dataset.threshold)));
  const active = ["queued", "running"].includes(job.status);
  $("#stopSaveButton").disabled = !active;
  $("#cancelButton").disabled = !active && job.status !== "stopping";
  $("#previewLabel").textContent = ["succeeded", "partial"].includes(job.status) ? "COMPLETE" : String(job.stage || "GENERATING").toUpperCase();
  const preview = $("#livePreview");
  let video = $("video", preview);
  if (["succeeded", "partial"].includes(job.status)) {
    if (!video) {
      video = document.createElement("video");
      video.muted = true; video.loop = true; video.autoplay = true; video.playsInline = true;
      preview.prepend(video);
    }
    if (!video.src) video.src = `/api/jobs/${job.id}/media`;
  } else if (video) video.remove();
}

async function refreshJobs() {
  try {
    const payload = await api("/api/jobs?limit=200");
    state.jobs = payload.jobs;
    renderJobs();
    const live = state.jobs.find((job) => ["running", "stopping", "cancelling", "queued"].includes(job.status));
    if (live && state.activeJobId !== live.id) {
      state.activeJobId = live.id;
      renderLive(live);
      connectEvents(live.id);
    } else if (state.activeJobId) {
      renderLive(state.jobs.find((job) => job.id === state.activeJobId) || null);
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderJobs() {
  const active = state.jobs.filter((job) => ["queued", "running", "stopping", "cancelling"].includes(job.status));
  $("#queueBadge").hidden = !active.length;
  $("#queueBadge").textContent = active.length;
  const queue = $("#queueList");
  const queueJobs = state.jobs.filter((job) => job.source === "webui" || active.includes(job));
  queue.innerHTML = queueJobs.length ? queueJobs.map((job, index) => jobRow(job, index)).join("") : '<div class="empty-list">キューは空です</div>';
  renderLibrary();
}

function jobRow(job, index) {
  const prompt = job.request?.prompt || "生成結果";
  const profile = job.request?.profile || job.plan?.request?.attention_backend || "history";
  const action = ["succeeded", "partial"].includes(job.status) ? "見る" : ["running", "queued", "stopping"].includes(job.status) ? "詳細" : "ログ";
  return `<article class="job-row" data-job="${job.id}">
    <div class="job-index">${String(index + 1).padStart(2,"0")}</div>
    <div class="job-main"><b>${escapeHTML(prompt)}</b><small>${escapeHTML(job.stage)} · ${escapeHTML(job.detail)}</small><div class="mini-progress"><progress max="100" value="${Number(job.progress || 0)}"></progress></div></div>
    <div class="job-stat"><span>STATUS</span><b>${statusLabels[job.status] || job.status}</b></div>
    <div class="job-stat"><span>PROFILE</span><b>${escapeHTML(profile)}</b></div>
    <button class="job-action" data-open-job="${job.id}">${action}</button>
  </article>`;
}

function renderLibrary() {
  const grid = $("#libraryGrid");
  const query = ($("#librarySearch")?.value || "").toLowerCase();
  const filter = $("#libraryFilter")?.value || "all";
  const jobs = state.jobs.filter((job) => {
    if (!["succeeded", "partial", "failed"].includes(job.status)) return false;
    if (filter !== "all" && job.status !== filter) return false;
    const haystack = `${job.request?.prompt || ""} ${job.output_path || ""}`.toLowerCase();
    return haystack.includes(query);
  });
  if (!jobs.length) {
    grid.innerHTML = '<div class="empty-library">条件に一致する生成履歴がありません</div>';
    return;
  }
  grid.innerHTML = jobs.map((job) => {
    const request = job.request || job.plan?.request || {};
    const success = ["succeeded", "partial"].includes(job.status);
    const playable = success && job.media_available !== false;
    const video = playable
      ? `<video src="/api/jobs/${job.id}/media#t=0.1" muted preload="metadata"></video>`
      : success ? "動画ファイルがありません" : "生成ログを確認";
    const frames = request.frames || "—";
    const size = request.width && request.height ? `${request.width}×${request.height}` : "—";
    return `<article class="library-card ${job.status}" data-open-job="${job.id}">
      <button type="button" class="library-card-delete" data-delete-job="${job.id}" aria-label="この生成を削除">削除</button>
      <div class="library-video">${video}</div>
      <div class="card-body"><div class="card-meta"><span>${statusLabels[job.status]}</span><time>${compactDate(job.created_at)}</time></div>
      <h3>${escapeHTML(request.prompt || "過去の生成結果")}</h3>
      <div class="card-specs"><span><b>${size}</b></span><span><b>${frames}</b> frames</span><span><b>${clock(job.duration_seconds)}</b></span></div></div>
    </article>`;
  }).join("");
}

function openJob(jobId) {
  const job = state.jobs.find((item) => item.id === jobId);
  if (!job) return;
  if (!["succeeded", "partial"].includes(job.status)) {
    state.activeJobId = job.id;
    renderLive(job);
    navigate("generate");
    return;
  }
  if (job.media_available === false) {
    toast("履歴は残っていますが、動画ファイルが見つかりません", "error");
    return;
  }
  const request = job.request || job.plan?.request || {};
  const dialog = $("#videoDialog");
  state.dialogJobId = job.id;
  state.dialogTrigger = document.activeElement;
  $("#dialogVideo").src = `/api/jobs/${job.id}/media`;
  $("#dialogBadge").textContent = statusLabels[job.status].toUpperCase();
  $("#dialogTitle").textContent = (job.output_path || "Generated video").split(/[\\/]/).pop();
  $("#dialogPrompt").textContent = request.prompt || "過去の生成結果";
  const peak = job.runtime_metrics?.cuda_peak_allocated_bytes;
  $("#dialogStats").innerHTML = `<div><small>TIME</small><b>${clock(job.duration_seconds)}</b></div><div><small>FRAMES</small><b>${request.frames || "—"}</b></div><div><small>VRAM PEAK</small><b>${bytes(peak)}</b></div>`;
  dialog.showModal();
}

function openDeleteDialog(jobId) {
  const job = state.jobs.find((item) => item.id === jobId);
  if (!job) return toast("削除する履歴が見つかりません", "error");
  if (!["succeeded", "partial", "failed", "cancelled", "interrupted"].includes(job.status)) {
    return toast("実行中または待機中の生成は削除できません", "error");
  }
  state.deleteJobId = job.id;
  state.deleteDialogTrigger = $("#videoDialog").open
    ? state.dialogTrigger
    : document.activeElement;
  const request = job.request || job.plan?.request || {};
  $("#deleteDialogPrompt").textContent = request.prompt || "過去の生成結果";
  $("#deleteDialogFilename").textContent = (job.output_path || "履歴のみ").split(/[\\/]/).pop();
  if ($("#videoDialog").open) $("#videoDialog").close();
  $("#deleteDialog").showModal();
}

async function deleteSelectedJob() {
  const jobId = state.deleteJobId;
  if (!jobId) return;
  const button = $("#confirmDeleteJob");
  button.disabled = true;
  button.textContent = "削除中…";
  try {
    const result = await api(`/api/jobs/${jobId}`, { method: "DELETE" });
    state.jobs = state.jobs.filter((job) => job.id !== jobId);
    if (state.activeJobId === jobId) state.activeJobId = null;
    $("#deleteDialog").close();
    renderJobs();
    const count = result.artifacts_deleted?.length || 0;
    toast(count ? `生成履歴と関連ファイル${count}件を削除しました` : "生成履歴を削除しました");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "完全に削除";
  }
}

async function stopActive(save) {
  if (!state.activeJobId) return;
  const path = save ? "stop-and-save" : "cancel";
  if (!save && !confirm("生成を今すぐ中止しますか？ 現在の映像は保存されない場合があります。")) return;
  try {
    const job = await api(`/api/jobs/${state.activeJobId}/${path}`, { method: "POST" });
    upsertJob(job);
    renderLive(job);
    toast(save ? "次のステップで停止して保存します" : "停止要求を送りました");
  } catch (error) { toast(error.message, "error"); }
}

function renderSettings(settings, readiness) {
  $("#settingConfig").value = settings.config_path || "";
  $("#settingUpstream").value = settings.upstream_path || "";
  $("#settingCheckpoint").value = settings.checkpoint_dir || "";
  $("#settingOutput").value = settings.output_dir || "";
  $("#settingPython").value = settings.python_path || "";
  $("#settingCache").value = settings.prompt_cache_dir || "";
  $$('[data-ready]').forEach((dot) => dot.classList.toggle("ready", Boolean(readiness?.[dot.dataset.ready]?.ready)));
  const ready = Object.values(readiness || {}).every((item) => item.ready);
  $("#engineState").textContent = ready ? "ENGINE READY" : "設定を確認";
  $(".engine-pill").classList.toggle("ready", ready);
}

async function saveSettings() {
  const payload = {
    config_path: $("#settingConfig").value.trim(),
    upstream_path: $("#settingUpstream").value.trim(),
    checkpoint_dir: $("#settingCheckpoint").value.trim(),
    output_dir: $("#settingOutput").value.trim(),
    python_path: $("#settingPython").value.trim(),
    prompt_cache_dir: $("#settingCache").value.trim(),
  };
  try {
    const result = await api("/api/settings", { method: "PUT", body: payload });
    renderSettings(result.settings, result.readiness);
    toast("エンジン設定を保存しました");
  } catch (error) { toast(error.message, "error"); }
}

function renderHardware(hardware) {
  const gpu = hardware?.gpus?.find((item) => item.selected_for_inference) || hardware?.gpus?.[0];
  if (gpu) $("#gpuName").textContent = gpu.name;
  $("#hardwarePanel").innerHTML = `<h3>Hardware</h3>
    <div class="hardware-item"><span>CPU</span><b>${escapeHTML(hardware?.cpu || "—")}</b></div>
    <div class="hardware-item"><span>Primary GPU</span><b>${escapeHTML(gpu?.name || "—")}<br>${bytes(gpu?.vram_total_bytes)}</b></div>
    <div class="hardware-item"><span>PyTorch / CUDA</span><b>${escapeHTML(hardware?.pytorch_version || "—")}<br>CUDA ${escapeHTML(hardware?.pytorch_cuda_version || "—")}</b></div>`;
}

async function pollResources() {
  try {
    const resources = await api("/api/system");
    const gpu = resources.gpus?.[0];
    if (gpu) {
      const percent = 100 * gpu.used_bytes / gpu.total_bytes;
      $("#vramText").textContent = `${bytes(gpu.used_bytes)} / ${bytes(gpu.total_bytes)}`;
      $("#vramMeter").value = percent;
      $("#gpuName").textContent = gpu.name;
    }
    $("#ramText").textContent = `${bytes(resources.ram_used_bytes)} / ${bytes(resources.ram_total_bytes)}`;
    $("#ramMeter").value = resources.ram_percent;
  } catch { /* keep the last good sample */ }
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
  $$('[data-go]').forEach((button) => button.addEventListener("click", () => navigate(button.dataset.go)));
  $("#generationForm").addEventListener("submit", submitGeneration);
  $("#prompt").addEventListener("input", () => $("#promptCount").textContent = $("#prompt").value.length);
  $("#structurePrompt").addEventListener("click", openPromptAssist);
  ["#promptVisual", "#promptSound", "#promptMusic"].forEach((selector) => {
    $(selector).addEventListener("input", structuredPromptDraft);
  });
  $("#applyStructuredPrompt").addEventListener("click", applyStructuredPrompt);
  $("#promptDialog").addEventListener("close", () => {
    state.dialogTrigger?.focus?.();
    state.dialogTrigger = null;
  });
  $$('[data-prompt-chip]').forEach((button) => button.addEventListener("click", () => {
    const prompt = $("#prompt"); prompt.value += `${prompt.value.trim() ? "\n\n" : ""}${button.dataset.promptChip}`; prompt.focus(); prompt.dispatchEvent(new Event("input"));
  }));
  $("#clearPrompt").addEventListener("click", () => { state.promptTransform = null; $("#prompt").value = ""; $("#prompt").dispatchEvent(new Event("input")); });
  $$('input[name="profile"]').forEach((radio) => radio.addEventListener("change", () => {
    $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(radio)));
    applyProfile(radio.value);
    if (radio.value === "custom") $("#advancedSettings").open = true;
  }));
  ["steps", "attention", "blocksSwap", "chunkRows", "vaeTile", "easycache", "ecThreshold", "ecStart", "ecEnd", "ecSkips"].forEach((id) => {
    $(`#${id}`).addEventListener("change", () => {
      const custom = $('input[name="profile"][value="custom"]');
      custom.checked = true;
      $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(custom)));
      updateProfileSummary();
    });
  });
  $$("#durationControl button").forEach((button) => button.addEventListener("click", () => {
    $$("#durationControl button").forEach((item) => item.classList.toggle("active", item === button));
    $("#customDuration").value = ""; state.duration = Number(button.dataset.duration); updateDuration();
  }));
  $("#customDuration").addEventListener("input", (event) => { if (event.target.value) { $$("#durationControl button").forEach((button) => button.classList.remove("active")); state.duration = Number(event.target.value); updateDuration(); } });
  $("#resolutionPreset").addEventListener("change", updateResolution);
  $("#width").addEventListener("input", updateResolution); $("#height").addEventListener("input", updateResolution);
  $("#randomSeed").addEventListener("click", () => $("#seed").value = Math.floor(Math.random() * 2147483647));
  $("#imageFile").addEventListener("change", (event) => uploadImage(event.target.files?.[0]));
  $("#removeImage").addEventListener("click", (event) => { event.preventDefault(); removeImage(); });
  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => uploadImage(event.dataTransfer.files?.[0]));
  $("#stopSaveButton").addEventListener("click", () => stopActive(true));
  $("#cancelButton").addEventListener("click", () => stopActive(false));
  $("#refreshButton").addEventListener("click", async () => { await Promise.all([refreshJobs(), pollResources()]); toast("最新情報に更新しました"); });
  $("#themeButton").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "clear" ? "dark" : "clear";
    applyTheme(next);
  });
  $("#saveSettings").addEventListener("click", saveSettings);
  $("#librarySearch").addEventListener("input", renderLibrary); $("#libraryFilter").addEventListener("change", renderLibrary);
  document.addEventListener("click", (event) => {
    const deleteTarget = event.target.closest("[data-delete-job]");
    if (deleteTarget) return openDeleteDialog(deleteTarget.dataset.deleteJob);
    const target = event.target.closest("[data-open-job]");
    if (target) openJob(target.dataset.openJob);
  });
  $("#deleteDialogJob").addEventListener("click", () => openDeleteDialog(state.dialogJobId));
  $("#confirmDeleteJob").addEventListener("click", deleteSelectedJob);
  $("#deleteDialog").addEventListener("close", () => {
    state.deleteJobId = null;
    if (state.deleteDialogTrigger?.isConnected) state.deleteDialogTrigger.focus?.();
    else $("#librarySearch").focus();
    state.deleteDialogTrigger = null;
  });
  $("#closeDialog").addEventListener("click", () => $("#videoDialog").close());
  $("#videoDialog").addEventListener("close", () => {
    $("#dialogVideo").pause();
    $("#dialogVideo").removeAttribute("src");
    state.dialogJobId = null;
    state.dialogTrigger?.focus?.();
    state.dialogTrigger = null;
  });
  document.addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter" && $("#view-generate").classList.contains("active")) submitGeneration(event); });
}

async function initialize() {
  applyTheme(savedTheme(), false);
  bindEvents();
  $("#prompt").dispatchEvent(new Event("input"));
  updateDuration(); updateResolution(); applyProfile("fast_sage");
  navigate(location.hash.slice(1) || "generate");
  try {
    const bootstrap = await api("/api/bootstrap");
    state.bootstrap = bootstrap;
    state.jobs = bootstrap.jobs || [];
    applyProfile(selectedProfile());
    $("#version").textContent = bootstrap.version;
    renderSettings(bootstrap.settings, bootstrap.readiness);
    renderHardware(bootstrap.hardware);
    renderJobs();
    const live = state.jobs.find((job) => ["running", "stopping", "cancelling", "queued"].includes(job.status));
    if (live) { state.activeJobId = live.id; renderLive(live); connectEvents(live.id); }
    else renderLive(null);
    await pollResources();
    state.resourceTimer = setInterval(pollResources, 3000);
    state.elapsedTimer = setInterval(() => { const job = state.jobs.find((item) => item.id === state.activeJobId); if (job) $("#elapsedTime").textContent = clock(activeElapsed(job)); }, 1000);
  } catch (error) {
    $("#engineState").textContent = "接続エラー";
    toast(error.message, "error");
  }
}

initialize();
