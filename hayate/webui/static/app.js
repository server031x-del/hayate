const SAMPLE_PROMPT = "A premium cinematic commercial for a sleek metallic silver sports car. The same car accelerates along a coastal highway at golden hour, dynamic tracking shots, close-ups of LED headlights and aerodynamic bodywork, then a final hero shot in a modern city plaza. Realistic motion, synchronized engine sound and cinematic music, no text, no logo, no watermark.";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  bootstrap: null,
  jobs: [],
  activeJobId: null,
  eventSource: null,
  eventSources: new Map(),
  duration: 5,
  imageAsset: null,
  lastImageAsset: null,
  dialogTrigger: null,
  resourceTimer: null,
  elapsedTimer: null,
  promptTransform: null,
  promptAI: null,
  modelSetup: null,
  fastH3Status: null,
  modelTimer: null,
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

// The model accepts 32-pixel geometry; these labels make the memory/time
// trade-off visible before a job is submitted.  They are intentionally
// hardware-neutral: the selected GPU and available RAM determine the actual
// operating point.
const RESOLUTION_PRESETS = {
  "512x512": { tier: "FAST", className: "fast", description: "速度を優先する安全な基準サイズ" },
  "512x288": { tier: "FAST", className: "fast", description: "横長SNS向け。軽量で試作に適した16:9" },
  "288x512": { tier: "FAST", className: "fast", description: "縦長SNS向け。軽量で試作に適した9:16" },
  "640x640": { tier: "STANDARD", className: "standard", description: "正方形の標準サイズ。速度と細部のバランス" },
  "768x448": { tier: "STANDARD", className: "standard", description: "横長の標準サイズ。5秒CMの基準におすすめ" },
  "864x480": { tier: "STANDARD", className: "standard", description: "ワイド画角の標準サイズ。映画的な構図向け" },
  "576x768": { tier: "DETAIL", className: "detail", description: "縦長の高精細。RAM/VRAM使用量が増えます" },
  "768x1024": { tier: "DETAIL", className: "detail", description: "縦長HD。長尺生成ではRAM/VRAMに注意" },
  "1024x576": { tier: "DETAIL", className: "detail", description: "16:9 HD。品質優先の書き出し向け" },
  "1280x736": { tier: "DETAIL", className: "detail", description: "HD映画サイズ。32GB RAMと十分な空き容量を推奨" },
  "1536x864": { tier: "DETAIL", className: "detail", description: "最高精細16:9。時間・メモリ負荷が最大です" },
  custom: { tier: "CUSTOM", className: "custom", description: "幅・高さを32の倍数で指定してください" },
};
const MODEL_ROLE_LABELS = {
  transformer: "DiT",
  text_encoder: "TEXT",
  video_vae: "VIDEO VAE",
  audio_vae: "AUDIO VAE",
  pdd_lora: "PDD LoRA",
  pdd_affine: "AdaLN",
  checkpoint_support: "SUPPORT",
};

function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) || "clear";
  } catch {
    return "clear";
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
  const checked = $('input[name="profile"]:checked');
  if (checked && !checked.disabled) return checked.value;
  const fallback = $$('input[name="profile"]').find((radio) => !radio.disabled);
  return fallback?.value || checked?.value || "fast_sage_detail";
}

function applyProfile(profile) {
  const radio = $(`input[name="profile"][value="${profile}"]`);
  if (radio?.disabled) {
    const fallback = $$('input[name="profile"]').find((item) => !item.disabled);
    if (fallback && fallback.value !== profile) {
      fallback.checked = true;
      $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(fallback)));
      return applyProfile(fallback.value);
    }
  }
  const preset = state.bootstrap?.profiles?.[profile];
  if (profile === "fasth3" || profile === "fasth3_fast") {
    // The preview is T2VA-only.  Keep the task selector aligned with the
    // backend contract while still letting the user switch back to a normal
    // profile for I2V/reference jobs.
    $("#task").value = "t2va";
  }
  if (preset) {
    $("#steps").value = preset.steps;
    $("#attention").value = preset.attention_backend;
    $("#easycache").checked = preset.easycache;
    $("#pdd").checked = Boolean(preset.pdd);
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
  updateComfyControls();
}

function updateComfyControls() {
  const comfy = selectedProfile() === "comfy_fasth3";
  $("#comfyOptions").hidden = !comfy;
  ["imageFile", "lastImageFile"].forEach(id => { $(`#${id}`).disabled = false; });
  $("#imageGuidance").textContent = comfy ? "FastH3でも開始・終了画像を指定できます。終了画像には開始画像が必要です。" : "画像なしでも生成できます。終了画像を使う場合は開始画像も選択してください。";
  ["vsaKeep", "fastVaeBatch"].forEach(id => { $(`#${id}`).disabled = !comfy; });
  $("#promptCache").disabled = comfy;
  if (comfy) {
    $("#profileAvailabilityNote").textContent = "FastH3 INT8の必要モデルと実行環境を準備済みです。";
    $("#profileAvailabilityNote").className = "model-availability-note ready";
    $("#profileAvailabilityNote").hidden = false;
    $("#advancedAvailabilityNote").hidden = true;
    $("#task").value = "t2va";
    $("#easycache").checked = false;
    $("#pdd").checked = false;
    ["task", "steps", "attention", "blocksSwap", "chunkRows", "vaeTile", "easycache", "pdd", "ecThreshold", "ecStart", "ecEnd", "ecSkips"].forEach(id => {
      setControlAvailability($(`#${id}`), false, "FastH3は専用設定を使用します");
    });
    $("#profileSummary").textContent = `FastH3 · 4-Step · VSA ${$("#vsaKeep").value}% · Fast VAE`;
  }
}

function updateProfileSummary() {
  if (selectedProfile() === "fasth3" || selectedProfile() === "fasth3_fast") {
    $("#profileSummary").textContent = selectedProfile() === "fasth3_fast"
      ? "FastH3 v1 Blackwell · sm100a/FA4/compile · 5 points（4-forward） · T2VAのみ"
      : "FastH3 VSA · 5 points（4-forward） · T2VAのみ";
    return;
  }
  const parts = [`${$("#steps").value} points`, $("#attention").value === "sageattn" ? "SageAttention" : "SDPA"];
  if ($("#easycache").checked) parts.push("EasyCache");
  if ($("#pdd").checked) parts.push("PDD 8-Step");
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
  const preset = $("#resolutionPreset").value;
  const custom = preset === "custom";
  $("#customResolution").hidden = !custom;
  const [width, height] = resolution();
  $("#resolutionHint").textContent = `${width} × ${height}`;
  const info = RESOLUTION_PRESETS[preset] || RESOLUTION_PRESETS.custom;
  const tier = $("#resolutionTier");
  tier.textContent = info.tier;
  tier.className = `resolution-tier ${info.className}`;
  $("#resolutionDescription").textContent = custom
    ? `${info.description}（現在 ${width} × ${height}）`
    : info.description;
}

async function uploadImage(file, last = false) {
  const prefix = last ? "lastImage" : "image";
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const dropzone = $(last ? "#lastDropzone" : "#dropzone");
  dropzone.classList.add("dragging");
  try {
    const asset = await api("/api/assets", { method: "POST", body: form });
    state[last ? "lastImageAsset" : "imageAsset"] = asset;
    $(`#${prefix}Name`).textContent = file.name;
    $(`#${prefix}Preview`).src = asset.url;
    $(`#${prefix}Preview`).hidden = false;
    $(last ? "#removeLastImage" : "#removeImage").hidden = false;
    $("#task").value = "auto";
    toast(`${last ? "終了" : "開始"}画像を読み込みました`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    dropzone.classList.remove("dragging");
  }
}

function removeImage(last = false) {
  const prefix = last ? "lastImage" : "image";
  state[last ? "lastImageAsset" : "imageAsset"] = null;
  $(`#${prefix}Name`).textContent = "画像未選択";
  $(`#${prefix}File`).value = "";
  $(`#${prefix}Preview`).removeAttribute("src");
  $(`#${prefix}Preview`).hidden = true;
  $(last ? "#removeLastImage" : "#removeImage").hidden = true;
}

function generationPayload() {
  const [width, height] = resolution();
  return {
    prompt: $("#prompt").value.trim(),
    original_prompt: state.promptTransform?.original || null,
    prompt_transform_applied: Boolean(state.promptTransform),
    prompt_transform_template_version: state.promptTransform?.version || null,
    profile: selectedProfile(),
    vsa_keep: Number($("#vsaKeep").value),
    fast_vae_batch: Number($("#fastVaeBatch").value),
    task: $("#task").value,
    width, height,
    duration_seconds: state.duration,
    seed: Number($("#seed").value),
    filename: "hayate",
    image_asset_id: state.imageAsset?.id || null,
    last_image_asset_id: state.lastImageAsset?.id || null,
    reference_asset_ids: [],
    use_prompt_cache: selectedProfile() !== "comfy_fasth3" && $("#promptCache").checked,
    steps: Number($("#steps").value),
    attention_backend: $("#attention").value,
    easycache: $("#easycache").checked,
    pdd: $("#pdd").checked,
    easycache_threshold: Number($("#ecThreshold").value),
    easycache_start: Number($("#ecStart").value),
    easycache_end: Number($("#ecEnd").value),
    easycache_max_consecutive_skips: Number($("#ecSkips").value),
    blocks_to_swap: Number($("#blocksSwap").value),
    activation_chunk_rows: Number($("#chunkRows").value),
    vae_tile_size: Number($("#vaeTile").value),
    gpu_device: $("#gpuDevice").value || "auto",
  };
}

function profileCanGenerate(profile = selectedProfile()) {
  if (profile === "comfy_fasth3") return state.modelSetup?.comfy_fasth3?.ready === true;
  const capabilities = modelCapabilities();
  if (profile === "fasth3" || profile === "fasth3_fast") {
    return profile === "fasth3_fast" ? capabilities.fastH3FastReady : capabilities.fastH3Ready;
  }
  return capabilities.known && capabilities.coreReady;
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

function setPromptAIStatus(message = "", kind = "") {
  const status = $("#promptAIStatus");
  status.textContent = message;
  status.classList.toggle("error", kind === "error");
}

function renderPromptAIResult(result) {
  const details = [
    `被写体: ${result.subject}`,
    `動作: ${result.action}`,
    `環境: ${result.environment}`,
    `カメラ: ${result.camera}`,
    `光・質感: ${result.lighting}`,
    `スタイル: ${result.style}`,
    `環境音・同期音: ${result.soundscape}`,
    `背景音楽: ${result.music}`,
    `ネガティブ確認: ${result.negative}`,
  ].join("\n");
  $("#promptAIResultText").textContent = details;
  $("#promptAIResult").hidden = false;
  $("#promptDraft").value = result.final_prompt;
}

async function generateAIPrompt() {
  const brief = $("#promptBrief").value.trim();
  if (!brief) return setPromptAIStatus("映像の概要を入力してください", "error");
  const [width, height] = resolution();
  const button = $("#generateAIPrompt");
  button.disabled = true;
  setPromptAIStatus("OpenAIでH3向け構成を作成中…");
  try {
    const result = await api("/api/prompt-assistant", {
      method: "POST",
      body: {
        brief,
        task: $("#promptTask").value,
        duration_seconds: state.duration,
        width,
        height,
        include_audio: $("#promptIncludeAudio").checked,
        language: "ja",
        current_prompt: state.promptTransform?.original || $("#prompt").value.trim(),
      },
    });
    state.promptAI = result;
    $("#promptVisual").value = `${result.subject}. ${result.action}. ${result.environment}. ${result.camera}. ${result.lighting}. ${result.style}.`;
    $("#promptSound").value = result.soundscape;
    $("#promptMusic").value = $("#promptIncludeAudio").checked ? result.music : "N/A";
    renderPromptAIResult(result);
    setPromptAIStatus(`AI案を作成しました（${state.bootstrap?.openai?.model || "OpenAI"}）。内容を確認して適用してください。`);
    toast("MiniMax H3向けAIプロンプト案を作成しました");
  } catch (error) {
    setPromptAIStatus(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function openPromptAssist() {
  const original = state.promptTransform?.original || $("#prompt").value.trim();
  state.promptAI = null;
  $("#promptBrief").value = original;
  $("#promptTask").value = $("#task").value;
  $("#promptIncludeAudio").checked = true;
  $("#promptAIResult").hidden = true;
  setPromptAIStatus("");
  $("#promptVisual").value = original;
  $("#promptSound").value = "";
  $("#promptMusic").value = "";
  structuredPromptDraft();
  state.dialogTrigger = document.activeElement;
  $("#promptDialog").showModal();
}

function applyStructuredPrompt() {
  const aiDraft = state.promptAI?.final_prompt || "";
  const keepAIDraft = Boolean(aiDraft && $("#promptDraft").value === aiDraft);
  const structured = keepAIDraft
    ? { draft: aiDraft, visual: $("#promptVisual").value.trim(), sound: $("#promptSound").value.trim(), music: $("#promptMusic").value.trim() }
    : structuredPromptDraft();
  const { draft, visual, sound, music } = structured;
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
  if (payload.last_image_asset_id && !payload.image_asset_id) return toast("終了画像を使う場合は開始画像も選択してください", "error");
  if (!payload.prompt) return toast("プロンプトを入力してください", "error");
  if (!profileCanGenerate(payload.profile)) {
    return toast("選択した構成のモデルが未取得または未検証です。設定画面でモデル状態を確認してください", "error");
  }
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
  state.eventSources.get(jobId)?.close();
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSources.set(jobId, source);
  state.eventSource = source;
  source.onmessage = (event) => {
    const job = JSON.parse(event.data);
    upsertJob(job);
    if (state.activeJobId === job.id) renderLive(job);
    renderJobs();
    if (["succeeded", "partial", "failed", "cancelled", "interrupted"].includes(job.status)) {
      source.close();
      state.eventSources.delete(job.id);
      if (["succeeded", "partial"].includes(job.status)) toast(job.status === "partial" ? "途中状態を保存しました" : "動画が完成しました");
      else if (job.status === "failed") toast(job.error || "生成に失敗しました", "error");
      refreshJobs();
    }
  };
  source.onerror = () => {
    source.close();
    state.eventSources.delete(jobId);
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
    .replace("pdd_sage", "PDD 8-Step + Sage")
    .replace("pdd", "PDD 8-Step")
    .replace("fast_sage_detail", "高速・画質優先")
    .replace("fast_sage", "速度優先")
    .replace("fast", "Fast SDPA")
    .replace("quality", "Quality");
  $$("#stageList span").forEach((span) => span.classList.toggle("done", progress >= Number(span.dataset.threshold)));
  const active = ["queued", "running"].includes(job.status);
  $("#stopSaveButton").disabled = !active || job.plan?.backend === "comfy_fasth3";
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
    const activeJobs = state.jobs.filter((job) => ["running", "stopping", "cancelling", "queued"].includes(job.status));
    activeJobs.forEach((job) => {
      if (!state.eventSources.has(job.id)) connectEvents(job.id);
    });
    for (const [jobId, source] of state.eventSources) {
      if (!activeJobs.some((job) => job.id === jobId)) {
        source.close();
        state.eventSources.delete(jobId);
      }
    }
    const live = activeJobs[0];
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
  const gpuLabel = job.assigned_gpu_name
    ? `GPU ${job.assigned_gpu_index ?? "?"} · ${job.assigned_gpu_name}`
    : (job.request?.effective_gpu_device || job.request?.gpu_device || "Auto");
  return `<article class="job-row" data-job="${job.id}">
    <div class="job-index">${String(index + 1).padStart(2,"0")}</div>
    <div class="job-main"><b>${escapeHTML(prompt)}</b><small>${escapeHTML(job.stage)} · ${escapeHTML(job.detail)}</small><div class="mini-progress"><progress max="100" value="${Number(job.progress || 0)}"></progress></div></div>
    <div class="job-stat"><span>STATUS</span><b>${statusLabels[job.status] || job.status}</b></div>
    <div class="job-stat"><span>GPU</span><b>${escapeHTML(gpuLabel)}</b></div>
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
    const haystack = `${job.request?.prompt || job.plan?.request?.prompt || ""} ${job.output_path || ""}`.toLowerCase();
    return haystack.includes(query);
  });
  $("#libraryCount").textContent = `${jobs.length}件`;
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
    const gpu = job.assigned_gpu_name ? `GPU ${job.assigned_gpu_index ?? "?"} · ${job.assigned_gpu_name}` : "Auto";
    return `<article class="library-card ${job.status}" data-open-job="${job.id}">
      <button type="button" class="library-card-delete" data-delete-job="${job.id}" aria-label="この生成を削除">削除</button>
      <div class="library-video">${video}<span class="preview-caption">${playable ? "クリックして拡大再生" : "ログ・詳細を確認"}</span></div>
      <div class="card-body"><div class="card-meta"><span>${statusLabels[job.status]}</span><time>${compactDate(job.created_at)}</time></div>
      <h3>${escapeHTML(request.prompt || "過去の生成結果")}</h3>
      <div class="card-specs"><span><b>${size}</b></span><span><b>${frames}</b> frames</span><span>生成時間 <b>${clock(job.duration_seconds)}</b></span><span title="${escapeHTML(gpu)}"><b>${escapeHTML(gpu)}</b></span></div><button type="button" class="secondary-button card-open" data-open-job="${job.id}">${playable ? "映像を再生・詳細を見る" : "生成の詳細を見る"}</button></div>
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
  $("#dialogStats").innerHTML = `<div><small>TIME</small>生成時間 <b>${clock(job.duration_seconds)}</b></div><div><small>FRAMES</small><b>${request.frames || "—"}</b></div><div><small>VRAM PEAK</small><b>${bytes(peak)}</b></div>`;
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

function renderSettings(settings, readiness, openai = state.bootstrap?.openai) {
  $("#settingConfig").value = settings.config_path || "";
  $("#settingUpstream").value = settings.upstream_path || "";
  $("#settingCheckpoint").value = settings.checkpoint_dir || "";
  $("#settingOutput").value = settings.output_dir || "";
  $("#settingPython").value = settings.python_path || "";
  $("#settingCache").value = settings.prompt_cache_dir || "";
  $("#settingPddCheckpoint").value = settings.pdd_checkpoint_path || "";
  $("#settingPddAffine").value = settings.pdd_adaln_affine_path || "";
  $("#settingFastVideoModel").value = settings.fastvideo_model_path || "";
  $("#settingFastVideoPython").value = settings.fastvideo_python_path || settings.python_path || "";
  renderGpuSelectors(state.bootstrap?.hardware, settings.gpu_default_selector || "auto");
  $("#settingGpuParallel").value = String(settings.gpu_parallel_jobs || 1);
  $("#settingOpenAIModel").value = openai?.model || "gpt-5.6-terra";
  // Never hydrate a secret back into the DOM.  The status only exposes the
  // configured/source state returned by the server.
  $("#settingOpenAIKey").value = "";
  $("#clearOpenAIKey").checked = false;
  const sourceLabels = { keyring: "Credential Manager", session: "この起動中", environment: "環境変数" };
  $("#openaiKeyStatus").textContent = openai?.api_key_configured
    ? `設定済み · ${sourceLabels[openai.api_key_source] || "保存済み"}`
    : "未設定";
  $$('[data-ready]').forEach((dot) => dot.classList.toggle("ready", Boolean(readiness?.[dot.dataset.ready]?.ready)));
  const ready = Object.values(readiness || {}).every((item) => item.ready);
  $("#engineState").textContent = ready ? "ENGINE READY" : "設定を確認";
  $(".engine-pill").classList.toggle("ready", ready);
}

function renderGpuSelectors(hardware, defaultSelector = "auto") {
  const devices = Array.isArray(hardware?.gpus) ? hardware.gpus : [];
  const options = [
    { value: "auto", label: "Auto · 空きGPUを選択" },
    ...devices.map((gpu) => ({
      value: gpu.uuid || String(gpu.index),
      label: gpu.h3_eligible
        ? (gpu.auto_assignable === false
          ? `GPU ${gpu.index} · ${gpu.name} · index指定のみ（UUIDなし）`
          : `GPU ${gpu.index} · ${gpu.name} · ${bytes(gpu.vram_total_bytes)}`)
        : `GPU ${gpu.index} · ${gpu.name} · 対象外（${gpu.eligibility_reason || "H3非対応"}）`,
      disabled: gpu.h3_eligible === false,
    })),
  ];
  [$("#gpuDevice"), $("#settingGpuDefault")].forEach((select) => {
    if (!select) return;
    const wanted = defaultSelector;
    select.innerHTML = options.map((option) => `<option value="${escapeHTML(option.value)}"${option.disabled ? " disabled" : ""}>${escapeHTML(option.label)}</option>`).join("");
    select.value = options.some((option) => option.value === wanted) ? wanted : "auto";
  });
}

function modelAssetStatus(asset) {
  if (asset.status === "downloading" || asset.download_status === "downloading") return "downloading";
  if (asset.status === "invalid") return "invalid";
  if (asset.status === "error" || asset.download_status === "error") return "retry";
  if (asset.status === "verified" || asset.verified) return "ready";
  if (asset.status === "present_unverified") return "present";
  if (asset.exists) return "present";
  return "missing";
}

function modelAssetStatusLabel(asset, status) {
  if (status === "ready") {
    if (asset.id === "transformer_fastvideo_vsa_4step") return "検証済み · FastH3 INT8で使用";
    if (asset.execution_supported === false) return "検証済み・HAYATE生成は未対応（外部ランタイム用）";
    if (asset.experimental) return "検証済み・実験経路（別ランタイム）";
    return asset.verified || asset.status === "verified" ? "検証済み・使用可能" : "ファイルあり（未検証）";
  }
  if (status === "downloading") {
    const progress = asset.progress ?? asset.download_progress;
    return Number.isFinite(Number(progress)) ? `ダウンロード中… ${Math.round(Number(progress))}%` : "ダウンロード中…";
  }
  if (status === "invalid") return "既存ファイルのサイズが一致しません。削除・配置を確認してください";
  if (status === "retry") return asset.error || asset.message || "取得に失敗しました。再試行できます";
  if (status === "present") {
    if (asset.id === "transformer_fastvideo_vsa_4step") return "ファイルあり · 検証完了後にFastH3 INT8で使用できます";
    if (asset.execution_supported === false) return "ファイルあり・ヘッダー診断と完全性検証のみ（HAYATE生成は未対応）";
    return asset.experimental ? "ファイルあり・FastH3診断が必要です" : "ファイルあり・SHA-256検証が必要です";
  }
  return asset.downloadable === false ? "未配置・公開元を確認して手動配置" : "未配置";
}

const STANDARD_MODEL_IDS = ["transformer_w4a8", "text_encoder_nvfp4_awq", "video_vae_int8_convrot", "audio_vae_fp32", "checkpoint_support"];
const PDD_MODEL_IDS = ["pdd_fl2va_8step", "pdd_adaln_affine"];
const COMFY_MODEL_IDS = ["transformer_fastvideo_vsa_4step", "text_encoder_nvfp4_awq", "video_vae_int8_convrot", "audio_vae_fp32"];
const MODEL_ASSET_SHORT_LABELS = {
  transformer_w4a8: "DiT",
  text_encoder_nvfp4_awq: "TEXT",
  video_vae_int8_convrot: "VIDEO VAE",
  audio_vae_fp32: "AUDIO VAE",
  checkpoint_support: "support files",
  pdd_fl2va_8step: "PDD LoRA",
  pdd_adaln_affine: "AdaLN",
};

function modelCapabilities() {
  const assets = Array.isArray(state.modelSetup?.assets) ? state.modelSetup.assets : [];
  const known = assets.length > 0;
  const byId = new Map(assets.map((asset) => [asset.id, asset]));
  const isReady = (id) => {
    const asset = byId.get(id);
    return Boolean(asset && asset.execution_supported !== false && modelAssetStatus(asset) === "ready");
  };
  const missing = (ids) => ids.filter((id) => !isReady(id));
  const coreMissing = missing(STANDARD_MODEL_IDS);
  const pddMissing = missing(PDD_MODEL_IDS);
  return {
    known,
    coreReady: known && coreMissing.length === 0,
    pddReady: known && coreMissing.length === 0 && pddMissing.length === 0,
    coreMissing,
    pddMissing,
    fastH3Ready: state.fastH3Status?.ready === true,
    fastH3FastReady: state.fastH3Status?.fast_profile_ready === true,
  };
}

function modelMissingLabel(ids) {
  return ids.map((id) => MODEL_ASSET_SHORT_LABELS[id] || id).join("、");
}

function setProfileAvailability(profile, enabled, reason = "") {
  const radio = $(`input[name="profile"][value="${profile}"]`);
  if (!radio) return;
  const card = radio.closest(".profile-card");
  radio.disabled = !enabled;
  card?.classList.toggle("unavailable", !enabled);
  card?.setAttribute("aria-disabled", String(!enabled));
  if (card) {
    let note = $(".profile-lock-note", card);
    if (!note) {
      note = document.createElement("small");
      note.className = "profile-lock-note";
      card.append(note);
    }
    note.textContent = reason;
    note.hidden = enabled || !reason;
    if (enabled) card.removeAttribute("title");
    else if (reason) card.title = reason;
  }
}

function setControlAvailability(control, enabled, reason = "") {
  if (!control) return;
  control.disabled = !enabled;
  control.setAttribute("aria-disabled", String(!enabled));
  const wrapper = control.closest("label") || control.closest(".easycache-box");
  wrapper?.classList.toggle("availability-disabled", !enabled);
  if (enabled) control.removeAttribute("title");
  else if (reason) control.title = reason;
}

function updateModelAvailability() {
  const capabilities = modelCapabilities();
  const profileNote = $("#profileAvailabilityNote");
  const advancedNote = $("#advancedAvailabilityNote");
  const standardReason = capabilities.known
    ? `標準H3の未取得または未検証: ${modelMissingLabel(capabilities.coreMissing)}`
    : "モデル取得状況を確認中です";
  const pddReason = capabilities.known
    ? `PDD追加モデルの未取得または未検証: ${modelMissingLabel(capabilities.pddMissing)}`
    : "モデル取得状況を確認中です";

  const standardEnabled = capabilities.coreReady;
  const comfyStatus = state.modelSetup?.comfy_fasth3;
  setProfileAvailability("comfy_fasth3", comfyStatus?.ready === true, comfyStatus?.message || "モデル・実行環境を確認中");
  $("#comfyReadiness").textContent = comfyStatus?.message || "FastH3環境は未確認です。再スキャンしてください";
  ["fast_sage", "fast_sage_detail", "fast", "quality", "custom"].forEach((profile) => {
    setProfileAvailability(profile, standardEnabled, standardEnabled ? "" : standardReason);
  });
  setProfileAvailability("pdd", capabilities.pddReady, capabilities.pddReady ? "" : capabilities.coreReady ? pddReason : standardReason);
  setProfileAvailability(
    "fasth3",
    capabilities.fastH3Ready,
    capabilities.fastH3Ready ? "" : (state.fastH3Status ? "FastH3の利用条件を満たしていません" : "FastH3の利用条件を診断してください"),
  );
  setProfileAvailability(
    "fasth3_fast",
    capabilities.fastH3FastReady,
    capabilities.fastH3FastReady ? "" : "Blackwell向けFastH3の利用条件を満たしていません",
  );
  const fastProfile = $('input[name="profile"][value="fasth3_fast"]');
  fastProfile?.closest(".profile-card")?.toggleAttribute("hidden", !capabilities.fastH3FastReady);

  if (profileNote) {
    if (!capabilities.known) {
      profileNote.hidden = true;
    } else {
      profileNote.hidden = false;
      profileNote.className = `model-availability-note ${capabilities.coreReady ? "ready" : "blocked"}`;
      profileNote.textContent = capabilities.coreReady
        ? (capabilities.pddReady
          ? "標準H3とPDD 8-Stepの必要モデルを検証済みです。"
          : "標準H3の必要モデルを検証済みです。PDD 8-Stepは追加モデル取得後に有効になります。")
        : `標準H3プロファイルは無効です。${standardReason}`;
    }
  }

  // Sampler, task, memory and cache controls depend on the standard H3
  // checkpoint. Seed, prompt cache and GPU selection remain usable because
  // they are independent of a particular model package.
  const dependentEnabled = capabilities.coreReady;
  const dependentReason = capabilities.coreReady ? "" : standardReason;
  ["steps", "attention", "blocksSwap", "chunkRows", "vaeTile", "easycache", "ecThreshold", "ecStart", "ecEnd", "ecSkips"].forEach((id) => {
    setControlAvailability($(`#${id}`), dependentEnabled, dependentReason);
  });
  const task = $("#task");
  setControlAvailability(task, dependentEnabled, dependentReason);
  $$("#task option").forEach((option) => {
    option.disabled = !dependentEnabled;
    option.title = dependentEnabled ? "" : dependentReason;
  });
  if (dependentEnabled && task?.value && $("#task option:checked")?.disabled) task.value = "auto";

  const pddEnabled = capabilities.pddReady;
  setControlAvailability($("#pdd"), pddEnabled, pddEnabled ? "" : pddReason);
  if (!pddEnabled && $("#pdd").checked) $("#pdd").checked = false;
  if (advancedNote) {
    if (!capabilities.known || capabilities.coreReady) {
      advancedNote.hidden = true;
    } else {
      advancedNote.hidden = false;
      advancedNote.className = "model-availability-note blocked";
      advancedNote.textContent = `モデル依存の詳細設定を無効化しています。${standardReason}`;
    }
  }

  const checked = $('input[name="profile"]:checked');
  if (!checked || checked.disabled) {
    const fallback = ["comfy_fasth3", "fast_sage_detail", "fast_sage", "fast", "quality", "pdd", "fasth3", "fasth3_fast", "custom"]
      .map((profile) => $(`input[name="profile"][value="${profile}"]`))
      .find((radio) => radio && !radio.disabled);
    if (fallback) {
      fallback.checked = true;
      $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(fallback)));
      applyProfile(fallback.value);
    }
  }
  updateProfileSummary();
  updateComfyControls();
}

function configurationAssets(assets) {
  const choice = $("#modelConfiguration").value;
  if (choice === "all") return assets;
  if (choice === "comfy") return assets.filter(asset => COMFY_MODEL_IDS.includes(asset.id));
  const ids = choice === "pdd" ? [...STANDARD_MODEL_IDS, "pdd_fl2va_8step", "pdd_adaln_affine"] : STANDARD_MODEL_IDS;
  return assets.filter(asset => ids.includes(asset.id));
}
let configurationDownloading = false;
async function downloadConfiguration() {
  if (configurationDownloading || !$("#modelLicenseConsent").checked || $("#modelConfiguration").value === "all") return;
  const assets = configurationAssets(state.modelSetup?.assets || []).filter(asset => asset.downloadable && !["ready", "invalid", "downloading"].includes(modelAssetStatus(asset)));
  configurationDownloading = true;
  renderModelSetup(state.modelSetup);
  try {
    await api("/api/models/setup/prepare", {method: "POST", body: {}});
    for (const asset of assets) {
      await api("/api/models/setup/download", {method: "POST", body: {asset_id: asset.id, license_accepted: true}});
    }
    toast("選択した構成の取得を受け付けました。各モデルの進捗を確認してください");
  } catch (error) { toast(error.message, "error"); }
  finally { configurationDownloading = false; await refreshModelSetup(true); }
}

function renderModelSetup(payload) {
  state.modelSetup = payload || {};
  updateModelAvailability();
  const allAssets = payload?.assets || payload?.models || [];
  const assets = configurationAssets(allAssets);
  const choice = $("#modelConfiguration").value;
  const total = assets.reduce((sum, asset) => sum + (Number(asset.size_bytes) || 0), 0);
  $("#modelConfigurationNote").textContent = choice === "all"
    ? "実験用・別エンジン用も含みます。すべてのモデルを取得する必要はありません。"
    : `${choice === "standard" ? "迷ったらこの構成。高速・画質優先で使う通常H3の必要セットです。" : "標準セットに8-Step用の追加モデルを含みます。品質は標準構成と比較してください。"} 必要ファイル ${assets.length}件・合計 ${bytes(total)}（VRAM必要量ではありません）。取得後は「標準パスを適用」を押してください。`;
  $("#downloadConfiguration").disabled = configurationDownloading || choice === "all" || !$("#modelLicenseConsent").checked || !assets.some(asset => asset.downloadable && !["ready", "invalid", "downloading"].includes(modelAssetStatus(asset)));
  if (choice === "comfy") {
    const remaining = assets.filter(a => modelAssetStatus(a) !== "ready").reduce((sum, a) => sum + Number(a.size_bytes || 0), 0);
    $("#modelConfigurationNote").textContent = `FastH3用4ファイル · 合計 ${bytes(total)} · 未準備 ${bytes(remaining)}。取得済みのTEXT・VAEを共有します。取得後はFastH3 INT8を選択してください。`;
  }

  const ready = assets.filter((asset) => modelAssetStatus(asset) === "ready").length;
  const activeCount = assets.filter((asset) => modelAssetStatus(asset) === "downloading").length;
  $("#modelSetupSummary").textContent = assets.length
    ? `${ready} / ${assets.length} 準備済み${activeCount ? ` · ${activeCount}件ダウンロード中` : ""}`
    : "モデル未確認";
  const list = $("#modelAssetList");
  if (!assets.length) {
    list.innerHTML = '<div class="model-setup-empty">モデルカタログを読み込めませんでした。再スキャンしてください。</div>';
    return;
  }
  const consent = $("#modelLicenseConsent").checked;
  list.innerHTML = assets.map((asset) => {
    const status = modelAssetStatus(asset);
    const size = asset.size_label || (asset.size_bytes ? bytes(asset.size_bytes) : "容量不明");
    const role = asset.role_label || MODEL_ROLE_LABELS[asset.role] || asset.role || "H3";
    const filename = asset.filename || asset.name || asset.label || asset.id;
    const path = asset.path || asset.relative_path || "標準フォルダ";
    const sourceUrl = typeof asset.source_url === "string" && asset.source_url.startsWith("https://")
      ? asset.source_url : "";
    const licenseUrl = typeof asset.license_url === "string" && asset.license_url.startsWith("https://")
      ? asset.license_url : "";
    const provenance = [
      sourceUrl ? `<a class="model-source" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noreferrer">公開元</a>` : "",
      licenseUrl ? `<a class="model-source" href="${escapeHTML(licenseUrl)}" target="_blank" rel="noreferrer">規約</a>` : "",
    ].filter(Boolean).join(" · ");
    const canDownload = Boolean(asset.downloadable) && status !== "ready" && status !== "invalid" && status !== "downloading" && consent;
    const actionLabel = status === "ready" ? "準備済み" : status === "present" ? "検証" : status === "downloading" ? "取得中…" : status === "invalid" ? "要確認" : status === "retry" ? "再試行" : asset.downloadable === false ? "手動配置" : "ダウンロード";
    const progress = status === "downloading" && Number.isFinite(Number(asset.progress ?? asset.download_progress))
      ? `<div class="model-progress"><span style="width:${Math.max(0, Math.min(100, Number(asset.progress ?? asset.download_progress)))}%"></span></div>` : "";
    const elapsed = asset.setup_elapsed_seconds;
    const timing = elapsed != null ? `${asset.download_finished_at ? "取得・検証時間" : "経過"} ${clock(elapsed)}` : "所要時間の記録なし";
    const speed = status === "downloading" && asset.download_speed_bytes_per_sec > 0 ? ` · ${bytes(asset.download_speed_bytes_per_sec)}/s` : "";
    const eta = status === "downloading" && asset.download_eta_seconds != null ? ` · 残り約 ${clock(asset.download_eta_seconds)}` : "";
    const downloadDetail = `<span class="model-download-detail">${escapeHTML(timing + speed + eta)}</span>`;
    const notes = Array.isArray(asset.notes) && asset.notes.length
      ? `<span class="model-asset-note">${escapeHTML(asset.notes.join(" / "))}</span>` : "";
    const experimental = asset.experimental ? `<span class="model-asset-experimental">EXPERIMENTAL</span>` : "";
    return `<article class="model-asset${asset.experimental ? " experimental" : ""}" data-model-id="${escapeHTML(asset.id || "")}">
      <div class="model-asset-main">
        <div class="model-asset-title"><span class="model-asset-role">${escapeHTML(role)}</span>${experimental}<b title="${escapeHTML(filename)}">${escapeHTML(asset.label || filename)}</b></div>
        <span class="model-asset-meta" title="${escapeHTML(path)}">${escapeHTML(filename)} · ${escapeHTML(size)}${provenance ? ` · ${provenance}` : ""}</span>
        <span class="model-asset-status ${status}">${escapeHTML(modelAssetStatusLabel(asset, status))}</span>${downloadDetail}${notes}${progress}
      </div>
      <button type="button" class="model-asset-action" data-model-download="${escapeHTML(asset.id || "")}" ${canDownload ? "" : "disabled"}>${actionLabel}</button>
    </article>`;
  }).join("");
  const active = allAssets.some((asset) => modelAssetStatus(asset) === "downloading");
  if (active && !state.modelTimer) {
    state.modelTimer = setInterval(() => refreshModelSetup(true), 2000);
  } else if (!active && state.modelTimer) {
    clearInterval(state.modelTimer);
    state.modelTimer = null;
  }
}

function renderFastH3Status(payload) {
  state.fastH3Status = payload || null;
  const element = $("#fastH3Status");
  if (!element) return;
  if (!payload) {
    element.className = "fasth3-status";
    element.textContent = "未診断";
    return;
  }
  const lines = [];
  const checkpoint = payload.checkpoint || {};
  if (checkpoint.exists) {
    lines.push(`チェックポイント: ${checkpoint.detected ? "VSA形式を検出" : "形式を確認できません"} · ${bytes(checkpoint.file_size)}`);
  } else {
    lines.push("チェックポイント: 未配置（ダウンロードは手動で開始できます）");
  }
  const runtime = payload.runtime || {};
  lines.push(`FastVideo runtime: ${runtime.available ? "検出済み" : "未検出"}`);
  if (Array.isArray(runtime.gpu_names) && runtime.gpu_names.length) {
    const caps = Array.isArray(runtime.gpu_capabilities) ? runtime.gpu_capabilities : [];
    lines.push(`GPU: ${runtime.gpu_names.map((name, index) => `${name}${caps[index] ? ` (sm_${String(caps[index]).replace(".", "")})` : ""}`).join(" / ")}`);
  }
  if (runtime.system_memory_bytes) lines.push(`WSL RAM: ${bytes(runtime.system_memory_bytes)}`);
  if (payload.model_directory) lines.push(`モデルフォルダ: ${payload.model_directory}`);
  if (payload.fastvideo_source_url) lines.push(`公式モデル: ${payload.fastvideo_source_url}`);
  if (payload.fastvideo_revision) lines.push(`固定revision: ${payload.fastvideo_revision}`);
  if (checkpoint.identity_verified === true) lines.push("Kijai単一ファイル: サイズ・構造・SHA-256を検証済み（外部ComfyUI用）");
  else if (checkpoint.exists) lines.push("Kijai単一ファイル: 構造検出済みですが、HAYATE FastVideo入力には使用しません");
  const checkpointIssues = Array.isArray(payload.checkpoint_issues) ? payload.checkpoint_issues : [];
  if (checkpointIssues.length) lines.push(...checkpointIssues.map((item) => `・単一ファイル診断: ${item}`));
  const issues = Array.isArray(payload.blocking_issues)
    ? payload.blocking_issues
    : (Array.isArray(payload.issues) ? payload.issues : []);
  if (issues.length) lines.push(...issues.map((item) => `・${item}`));
  else lines.push("公式FastVideoディレクトリとランタイムが利用可能です。FastH3専用経路を選択できます。");
  const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];
  if (warnings.length) lines.push(...warnings.map((item) => `・注意: ${item}`));
  element.className = `fasth3-status ${payload.ready ? "ready" : "blocked"}`;
  element.textContent = lines.join("\n");
  if (Array.isArray(payload.fast_profile_issues) && payload.fast_profile_issues.length) {
    lines.push(...payload.fast_profile_issues.map((item) => `・Blackwell最速条件: ${item}`));
  } else if (payload.fast_profile_ready === true) {
    lines.push("Blackwell最速プロファイル: 利用可能（sm100a VSA + FA4 + regional compile）");
  }
  element.textContent = lines.join("\n");
  updateModelAvailability();
}

async function checkFastH3() {
  const button = $("#checkFastH3");
  if (!button) return;
  button.disabled = true;
  button.textContent = "診断中…";
  try {
    const payload = await api("/api/models/fasth3");
    renderFastH3Status(payload);
    toast(payload.ready ? "FastH3の利用条件を満たしています" : "FastH3はまだ利用条件を満たしていません", payload.ready ? "success" : "error");
  } catch (error) {
    renderFastH3Status({ ready: false, issues: [error.message] });
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "利用条件を診断";
  }
}

async function refreshModelSetup(silent = false) {
  try {
    const payload = await api("/api/models/setup");
    renderModelSetup(payload);
  } catch (error) {
    $("#modelSetupSummary").textContent = "確認できません";
    state.modelSetup = null;
    updateModelAvailability();
    $("#modelAssetList").innerHTML = `<div class="model-setup-empty">${escapeHTML(error.message)}</div>`;
    if (!silent) toast(error.message, "error");
  }
}

async function prepareModelDirs() {
  const button = $("#prepareModelDirs");
  button.disabled = true;
  try {
    const result = await api("/api/models/setup/prepare", { method: "POST", body: {} });
    if (result.settings && state.bootstrap) {
      state.bootstrap.settings = result.settings;
      renderSettings(result.settings, result.readiness, result.openai);
    }
    await refreshModelSetup(true);
    toast("標準モデルフォルダを準備しました");
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = false; }
}

async function applyStandardPaths() {
  if (!confirm("HAYATE標準のモデルパスを設定に反映しますか？\n現在のカスタム checkpoint / PDD パスは置き換わります。")) return;
  const button = $("#applyStandardPaths");
  button.disabled = true;
  try {
    const result = await api("/api/models/setup/apply-standard", { method: "POST", body: {} });
    if (state.bootstrap) state.bootstrap.settings = result.settings;
    renderSettings(result.settings, result.readiness, result.openai);
    toast("標準モデルパスを設定に反映しました");
    await refreshModelSetup(true);
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = false; }
}

async function downloadModel(assetId) {
  if (!$("#modelLicenseConsent").checked) {
    toast("ライセンス確認にチェックを入れてからダウンロードしてください", "error");
    return;
  }
  const button = $$("[data-model-download]").find((item) => item.dataset.modelDownload === assetId);
  if (button) { button.disabled = true; button.textContent = "準備中…"; }
  try {
    await api("/api/models/setup/download", {
      method: "POST",
      body: { asset_id: assetId, license_accepted: true },
    });
    await refreshModelSetup(true);
    toast("モデルのダウンロードを開始しました。設定画面で進捗を確認できます");
  } catch (error) {
    if (button) { button.disabled = false; button.textContent = "ダウンロード"; }
    toast(error.message, "error");
  }
}

async function saveSettings() {
  const payload = {
    config_path: $("#settingConfig").value.trim(),
    upstream_path: $("#settingUpstream").value.trim(),
    checkpoint_dir: $("#settingCheckpoint").value.trim(),
    output_dir: $("#settingOutput").value.trim(),
    python_path: $("#settingPython").value.trim(),
    prompt_cache_dir: $("#settingCache").value.trim(),
    pdd_checkpoint_path: $("#settingPddCheckpoint").value.trim(),
    pdd_adaln_affine_path: $("#settingPddAffine").value.trim(),
    fastvideo_model_path: $("#settingFastVideoModel").value.trim(),
    fastvideo_python_path: $("#settingFastVideoPython").value.trim(),
    gpu_default_selector: $("#settingGpuDefault").value || "auto",
    gpu_parallel_jobs: Number($("#settingGpuParallel").value || 1),
    openai_model: $("#settingOpenAIModel").value.trim(),
    openai_api_key: $("#settingOpenAIKey").value.trim() || null,
    clear_openai_api_key: $("#clearOpenAIKey").checked,
  };
  try {
    const result = await api("/api/settings", { method: "PUT", body: payload });
    if (state.bootstrap) state.bootstrap.openai = result.openai;
    renderSettings(result.settings, result.readiness, result.openai);
    toast("エンジン設定を保存しました");
  } catch (error) { toast(error.message, "error"); }
}

function renderHardware(hardware) {
  const gpu = hardware?.gpus?.find((item) => item.selected_for_inference) || hardware?.gpus?.[0];
  if (gpu) $("#gpuName").textContent = gpu.name;
  renderGpuSelectors(hardware, state.bootstrap?.settings?.gpu_default_selector || "auto");
  const gpuRows = (hardware?.gpus || []).map((item) => `<div class="hardware-item"><span>GPU ${item.index}${item.selected_for_inference ? " · primary" : ""}</span><b>${escapeHTML(item.name)}<br>${bytes(item.vram_total_bytes)}${item.uuid ? `<br><small>${escapeHTML(item.uuid)}</small>` : ""}</b></div>`).join("");
  $("#hardwarePanel").innerHTML = `<h3>Hardware</h3>
    <div class="hardware-item"><span>CPU</span><b>${escapeHTML(hardware?.cpu || "—")}</b></div>
    ${gpuRows || `<div class="hardware-item"><span>GPU</span><b>検出できません</b></div>`}
    <div class="hardware-item"><span>GPU scheduling</span><b>${hardware?.auto_assignable_gpu_count ?? hardware?.eligible_gpu_count ?? 0} / ${hardware?.gpu_count || 0} auto-assignable<br>${hardware?.multi_gpu_supported ? "multi-GPU scheduling available" : "single-adapter"}</b></div>
    <div class="hardware-item"><span>PyTorch / CUDA</span><b>${escapeHTML(hardware?.pytorch_version || "—")}<br>CUDA ${escapeHTML(hardware?.pytorch_cuda_version || "—")}</b></div>`;
}

async function pollResources() {
  try {
    const resources = await api("/api/system");
    const activeJob = state.jobs.find((job) => job.id === state.activeJobId);
    const gpu = resources.gpus?.find((item) => item.index === activeJob?.assigned_gpu_index)
      || resources.gpus?.[0];
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
  $("#generateAIPrompt").addEventListener("click", generateAIPrompt);
  $("#promptDialog").addEventListener("close", () => {
    state.dialogTrigger?.focus?.();
    state.dialogTrigger = null;
  });
  $$('[data-prompt-chip]').forEach((button) => button.addEventListener("click", () => {
    const prompt = $("#prompt"); prompt.value += `${prompt.value.trim() ? "\n\n" : ""}${button.dataset.promptChip}`; prompt.focus(); prompt.dispatchEvent(new Event("input"));
  }));
  $("#samplePrompt").addEventListener("click", () => { state.promptTransform = null; $("#prompt").value = SAMPLE_PROMPT; $("#prompt").dispatchEvent(new Event("input")); $("#prompt").focus(); });
  $("#clearPrompt").addEventListener("click", () => { state.promptTransform = null; $("#prompt").value = ""; $("#prompt").dispatchEvent(new Event("input")); });
  $$('input[name="profile"]').forEach((radio) => radio.addEventListener("change", () => {
    $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(radio)));
    applyProfile(radio.value);
    updateModelAvailability();
    if (radio.value === "custom") $("#advancedSettings").open = true;
  }));
  ["steps", "attention", "blocksSwap", "chunkRows", "vaeTile", "easycache", "pdd", "ecThreshold", "ecStart", "ecEnd", "ecSkips"].forEach((id) => {
    $(`#${id}`).addEventListener("change", () => {
      const custom = $('input[name="profile"][value="custom"]');
      custom.checked = true;
      $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(custom)));
      updateProfileSummary();
    });
  });
  $("#easycache").addEventListener("change", () => {
    if ($("#easycache").checked) $("#pdd").checked = false;
    updateProfileSummary();
  });
  ["vsaKeep", "fastVaeBatch"].forEach(id => $(`#${id}`).addEventListener("change", updateComfyControls));
  $("#pdd").addEventListener("change", () => {
    if ($("#pdd").checked) {
      $("#easycache").checked = false;
      $("#steps").value = 9;
    }
    updateProfileSummary();
  });
  $$("#durationControl button").forEach((button) => button.addEventListener("click", () => {
    $$("#durationControl button").forEach((item) => item.classList.toggle("active", item === button));
    $("#customDuration").value = ""; state.duration = Number(button.dataset.duration); updateDuration();
  }));
  $("#customDuration").addEventListener("input", (event) => { if (event.target.value) { $$("#durationControl button").forEach((button) => button.classList.remove("active")); state.duration = Number(event.target.value); updateDuration(); } });
  $("#resolutionPreset").addEventListener("change", updateResolution);
  $("#width").addEventListener("input", updateResolution); $("#height").addEventListener("input", updateResolution);
  $("#randomSeed").addEventListener("click", () => $("#seed").value = Math.floor(Math.random() * 2147483647));
  [false, true].forEach(last => {
    const prefix = last ? "lastImage" : "image";
    $(`#${prefix}File`).addEventListener("change", event => uploadImage(event.target.files?.[0], last));
    $(last ? "#removeLastImage" : "#removeImage").addEventListener("click", () => removeImage(last));
    const zone = $(last ? "#lastDropzone" : "#dropzone");
    ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove("dragging"); }));
    zone.addEventListener("drop", event => uploadImage(event.dataTransfer.files?.[0], last));
  });
  $("#stopSaveButton").addEventListener("click", () => stopActive(true));
  $("#cancelButton").addEventListener("click", () => stopActive(false));
  $("#refreshButton").addEventListener("click", async () => { await Promise.all([refreshJobs(), pollResources()]); toast("最新情報に更新しました"); });
  $("#themeButton").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "clear" ? "dark" : "clear";
    applyTheme(next);
  });
  $("#saveSettings").addEventListener("click", saveSettings);
  $("#prepareModelDirs").addEventListener("click", prepareModelDirs);
  $("#applyStandardPaths").addEventListener("click", applyStandardPaths);
  $("#refreshModelSetup").addEventListener("click", () => refreshModelSetup());
  $("#checkFastH3").addEventListener("click", checkFastH3);
  $("#modelConfiguration").addEventListener("change", () => renderModelSetup(state.modelSetup));
  $("#downloadConfiguration").addEventListener("click", downloadConfiguration);
  $("#modelLicenseConsent").addEventListener("change", () => renderModelSetup(state.modelSetup));
  $("#modelAssetList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-model-download]");
    if (button) downloadModel(button.dataset.modelDownload);
  });
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
  updateDuration(); updateResolution(); applyProfile("fast_sage_detail");
  navigate(location.hash.slice(1) || "generate");
  try {
    const bootstrap = await api("/api/bootstrap");
    state.bootstrap = bootstrap;
    state.jobs = bootstrap.jobs || [];
    applyProfile(selectedProfile());
    $("#version").textContent = bootstrap.version;
    renderSettings(bootstrap.settings, bootstrap.readiness, bootstrap.openai);
    renderModelSetup({ assets: [] });
    await refreshModelSetup(true);
    renderHardware(bootstrap.hardware);
    renderJobs();
    const activeJobs = state.jobs.filter((job) => ["running", "stopping", "cancelling", "queued"].includes(job.status));
    activeJobs.forEach((job) => connectEvents(job.id));
    const live = activeJobs[0];
    if (live) { state.activeJobId = live.id; renderLive(live); }
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
