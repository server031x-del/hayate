const SAMPLE_PROMPT = "A premium cinematic commercial for a sleek metallic silver sports car. The same car accelerates along a coastal highway at golden hour, dynamic tracking shots, close-ups of LED headlights and aerodynamic bodywork, then a final hero shot in a modern city plaza. Realistic motion, synchronized engine sound and cinematic music, no text, no logo, no watermark.";
const I2V_SAMPLE_PROMPT = "Use <Picture 1> as the exact opening frame. Keep the same character, facial features, clothing, colors, and original 2D anime illustration style throughout. The character blinks and moves gently while the camera makes a slow, subtle push-in. Preserve the original background and lighting. Do not turn the character into a photorealistic or live-action person. No text, no logo, no watermark.";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  bootstrap: null,
  bootstrapError: false,
  jobs: [],
  activeJobId: null,
  eventSources: new Map(),
  duration: 5,
  imageAsset: null,
  lastImageAsset: null,
  imageSizes: { first: null, last: null },
  dialogTrigger: null,
  dialogJobId: null,
  resourceTimer: null,
  elapsedTimer: null,
  resourceFailures: 0,
  resourcePolling: false,
  connectionLost: false,
  promptTransform: null,
  promptAI: null,
  modelSetup: null,
  modelSetupLoaded: false,
  modelSourceUrls: {},
  openModelSources: new Set(),
  fastH3Status: null,
  modelTimer: null,
  preferredProfile: null,
  submitting: false,
  libraryKey: "",
  recentKey: "",
  videoObserver: null,
  draftTimer: null,
  settingsDirty: false,
  view: "generate",
};

const viewMeta = {
  generate: ["CREATE", "新しい映像をつくる", "プロンプトと画像から、映像と音声を生成します"],
  queue: ["JOBS", "生成キュー", "実行中・待機中のジョブと最近の結果"],
  library: ["LIBRARY", "生成ライブラリ", "過去の映像、設定、実行統計をまとめて確認"],
  settings: ["SYSTEM", "エンジン設定", "モデルの準備、保存先、AIプロンプト作成"],
};
const VIEW_KEYS = { 1: "generate", 2: "queue", 3: "library", 4: "settings" };

const statusLabels = {
  queued: "待機中", running: "生成中", stopping: "途中保存中", cancelling: "停止中",
  succeeded: "完了", partial: "途中保存", failed: "失敗", cancelled: "中止", interrupted: "中断",
};
const ACTIVE_STATUSES = ["queued", "running", "stopping", "cancelling"];
const FINAL_STATUSES = ["succeeded", "partial", "failed", "cancelled", "interrupted"];
const PROFILE_LABELS = {
  comfy_fasth3: "FastH3 INT8",
  comfy_fl2va: "画像優先 FL2VA",
  fast_sage_detail: "高速・画質優先",
  fast_sage: "速度優先",
  quality: "Quality",
  pdd: "PDD 8-Step",
  pdd_sage: "PDD 8-Step + Sage",
  fast: "Fast SDPA",
  fasth3: "FastH3 VSA",
  fasth3_fast: "FastH3 Blackwell",
  custom: "Custom",
};
const SECONDARY_PROFILES = new Set(["pdd", "fast", "fasth3", "fasth3_fast", "custom"]);
const ADVANCED_CONTROL_IDS = ["steps", "attention", "blocksSwap", "chunkRows", "vaeTile", "easycache", "pdd", "ecThreshold", "ecStart", "ecEnd", "ecSkips"];
const FIELD_LABELS = {
  prompt: "プロンプト", width: "幅", height: "高さ", duration_seconds: "動画の長さ", seed: "Seed",
  steps: "Scheduler points", blocks_to_swap: "Block swap", activation_chunk_rows: "Activation chunk rows",
  easycache_threshold: "EasyCache threshold", easycache_start: "EasyCache start", easycache_end: "EasyCache end",
  easycache_max_consecutive_skips: "EasyCache max skips", vae_tile_size: "VAE tile", gpu_device: "実行GPU",
};

const THEME_KEY = "hayate-studio-theme";
const DRAFT_KEY = "hayate-studio-draft-v1";
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];

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
  custom: { tier: "CUSTOM", className: "custom", description: "幅・高さを32の倍数（256〜1536）で指定してください" },
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

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) || "clear";
  } catch {
    return "clear";
  }
}

function applyTheme(theme, persist = true) {
  const selected = theme === "dark" ? "dark" : "clear";
  document.documentElement.dataset.theme = selected;
  const button = $("#themeButton");
  if (button) {
    const clear = selected === "clear";
    button.setAttribute("aria-pressed", String(!clear));
    button.title = clear ? "Darkモードに切り替え" : "Clearモードに切り替え";
    button.setAttribute("aria-label", button.title);
    $("#themeLabel").textContent = clear ? "DARK" : "CLEAR";
  }
  document.querySelector('meta[name="theme-color"]')?.setAttribute(
    "content", selected === "clear" ? "#eef3f8" : "#080c13"
  );
  if (persist) {
    try { localStorage.setItem(THEME_KEY, selected); } catch { /* optional */ }
  }
}

function errorMessage(payload, status) {
  const detail = payload && typeof payload === "object" ? payload.detail : payload;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const lines = detail.map((item) => {
      const field = (item?.loc || []).filter((part) => part !== "body").join(".");
      return `・${FIELD_LABELS[field] || field || "入力"}: ${item?.msg || "不正な値です"}`;
    });
    return ["入力内容を確認してください", ...lines].join("\n");
  }
  if (detail && typeof detail === "object") {
    const issues = Array.isArray(detail.issues) ? detail.issues.map((issue) => `・${issue}`) : [];
    const head = detail.message === "generation preflight failed"
      ? "生成前チェックで問題が見つかりました"
      : (detail.message || "");
    const text = [head, ...issues].filter(Boolean).join("\n");
    if (text) return text;
  }
  if (status === 404) return "対象が見つかりません（HTTP 404）";
  if (status >= 500) return `サーバーでエラーが発生しました（HTTP ${status}）`;
  return `リクエストに失敗しました（HTTP ${status}）`;
}

async function api(path, options = {}) {
  const request = { ...options, headers: { ...(options.headers || {}) } };
  if (request.method && request.method !== "GET") request.headers["X-HAYATE-UI"] = "1";
  if (request.body && !(request.body instanceof FormData) && typeof request.body !== "string") {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(request.body);
  }
  let response;
  try {
    response = await fetch(path, request);
  } catch {
    throw new Error("HAYATEサーバーに接続できません。サーバーが起動しているか確認してください");
  }
  const type = response.headers.get("content-type") || "";
  let payload = null;
  try {
    payload = type.includes("json") ? await response.json() : await response.text();
  } catch {
    payload = null;
  }
  if (!response.ok) throw new Error(errorMessage(payload, response.status));
  return payload;
}

function toast(message, kind = "success", options = {}) {
  const stack = $("#toastStack");
  if (!stack) return;
  const element = document.createElement("div");
  element.className = `toast ${kind}`;
  element.setAttribute("role", kind === "error" ? "alert" : "status");
  const text = document.createElement("p");
  text.textContent = message;
  element.append(text);
  let timer = null;
  const dismiss = () => {
    if (!element.isConnected) return;
    element.classList.add("leaving");
    setTimeout(() => element.remove(), 200);
  };
  const schedule = () => {
    clearTimeout(timer);
    timer = setTimeout(dismiss, options.duration ?? (kind === "error" ? 9000 : 4500));
  };
  if (options.action) {
    const action = document.createElement("button");
    action.type = "button";
    action.textContent = options.action.label;
    action.addEventListener("click", () => { options.action.run(); dismiss(); });
    element.append(action);
  }
  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast-close";
  close.setAttribute("aria-label", "通知を閉じる");
  close.textContent = "×";
  close.addEventListener("click", dismiss);
  element.append(close);
  element.addEventListener("mouseenter", () => clearTimeout(timer));
  element.addEventListener("mouseleave", schedule);
  stack.append(element);
  while (stack.children.length > 4) stack.firstElementChild.remove();
  schedule();
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

function compactDate(value, long = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const options = long
    ? { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" };
  return new Intl.DateTimeFormat("ja-JP", options).format(date);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function framesFor(seconds) {
  return 17 * Math.max(1, Math.round((seconds * 24 - 5) / 17)) + 5;
}

function isActive(job) {
  return Boolean(job && ACTIVE_STATUSES.includes(job.status));
}

function findJob(jobId) {
  return state.jobs.find((item) => item.id === jobId) || null;
}

function jobRequest(job) {
  return job?.request || job?.plan?.request || {};
}

function jobPrompt(job) {
  const request = jobRequest(job);
  return request.effective_prompt || request.prompt || "";
}

function jobFilename(job) {
  return (job?.output_path || "").split(/[\\/]/).pop() || "";
}

function profileLabel(job) {
  const profile = jobRequest(job).profile;
  if (profile) return PROFILE_LABELS[profile] || profile;
  return job?.source === "history" ? "CLI" : "—";
}

function gpuLabel(job) {
  if (job?.assigned_gpu_name) return `GPU ${job.assigned_gpu_index ?? "?"} · ${job.assigned_gpu_name}`;
  const selector = jobRequest(job).effective_gpu_device || jobRequest(job).gpu_device;
  return !selector || selector === "auto" ? "Auto" : selector;
}

// H3 prompts are verbose structured text; show the visual description only.
function displayPrompt(text) {
  const raw = String(text || "").trim();
  const visual = raw
    .replace(/For the target video,[\s\S]*?referenced\.\s*/gi, "")
    .split(/\s*(?:overall_soundscape|non_diegetic_music|negative_prompt)\s*:/i)[0]
    .replace(/integrated_multimodal_description\s*:\s*/gi, "")
    .replace(/\[Shot[^\]]*\]\s*/gi, "")
    .replace(/\s+/g, " ")
    .trim();
  return visual || raw;
}

function hashView() {
  const view = location.hash.slice(1);
  return viewMeta[view] ? view : "generate";
}

/* ------------------------------------------------------------------ */
/* Navigation                                                          */
/* ------------------------------------------------------------------ */

function navigate(view) {
  const target = viewMeta[view] ? view : "generate";
  if (location.hash.slice(1) === target) showView(target);
  else location.hash = target;
}

function showView(view) {
  state.view = view;
  $$(".nav-item").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
  const [eyebrow, title, subtitle] = viewMeta[view];
  $("#viewEyebrow").textContent = eyebrow;
  $("#viewTitle").textContent = title;
  $("#viewSubtitle").textContent = subtitle;
  if (view === "queue" || view === "library") refreshJobs();
  window.scrollTo({ top: 0 });
}

/* ------------------------------------------------------------------ */
/* Profiles and model availability                                     */
/* ------------------------------------------------------------------ */

function profileRadio(profile) {
  return $(`input[name="profile"][value="${profile}"]`);
}

function checkProfile(radio) {
  if (!radio) return;
  radio.checked = true;
  $$(".profile-card").forEach((card) => card.classList.toggle("selected", card.contains(radio)));
  if (SECONDARY_PROFILES.has(radio.value)) {
    const more = $("#moreProfiles");
    if (more) more.open = true;
  }
}

function selectedProfile() {
  const checked = $('input[name="profile"]:checked');
  if (checked && !checked.disabled) return checked.value;
  const fallback = $$('input[name="profile"]').find((radio) => !radio.disabled);
  return fallback?.value || checked?.value || "fast_sage_detail";
}

function applyProfile(profile) {
  const radio = profileRadio(profile);
  if (radio?.disabled) {
    const fallback = $$('input[name="profile"]').find((item) => !item.disabled);
    if (fallback && fallback.value !== profile) {
      checkProfile(fallback);
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
  }
  updateProfileSummary();
  updateComfyControls();
}

function updateComfyControls() {
  const profile = selectedProfile();
  const comfy = profile === "comfy_fasth3" || profile === "comfy_fl2va";
  const imageMode = profile === "comfy_fl2va";
  const textOnly = profile === "comfy_fasth3" || profile === "fasth3" || profile === "fasth3_fast";
  $("#comfyOptions").hidden = !comfy;
  $("#comfySpeedOptions").hidden = imageMode;
  ["imageFile", "lastImageFile"].forEach((id) => { $(`#${id}`).disabled = textOnly; });
  $("#imageGuidance").textContent = textOnly
    ? "このプロファイルはテキストのみ（T2VA）に対応しています。画像を使う場合は別のプロファイルを選択してください。"
    : comfy
      ? "開始画像の画風と人物を引き継ぐFL2VAモデルを使います。画像比率を出力に合わせ、プロンプトにも同じ2D画風を明記してください。"
      : "画像なしでも生成できます。終了画像を使う場合は開始画像も選択してください。";
  ["vsaKeep", "fastVaeBatch"].forEach((id) => { $(`#${id}`).disabled = !comfy || imageMode; });
  $("#comfyOptionsTitle").textContent = imageMode ? "FL2VA · 画像から動画" : "FastH3 · 画質と速度";
  $("#comfyOptionsNote").textContent = imageMode
    ? "画像用のFL2VA重みで開始画像を条件にします。50ステップのためFastH3より時間がかかります。"
    : "テキストから動画＋音声を4ステップで生成します。画像は使えません。";
  $("#promptCache").disabled = comfy;
  if (comfy) {
    const note = $("#profileAvailabilityNote");
    const ready = state.modelSetup?.[profile]?.ready === true;
    note.textContent = ready
      ? (imageMode
        ? "FL2VAの必要モデルと実行環境を準備済みです。開始画像を選択してください。"
        : "FastH3 INT8はテキスト生成専用です。画像から生成する場合は『画像優先 FL2VA』を選んでください。")
      : (state.modelSetup?.[profile]?.message || "必要モデルと実行環境を確認中です");
    note.className = `availability-note ${ready ? "ready" : "blocked"}`;
    note.hidden = false;
    $("#advancedAvailabilityNote").hidden = true;
    $("#task").value = imageMode ? "fl2va" : "t2va";
    $("#easycache").checked = false;
    $("#pdd").checked = false;
    ["task", ...ADVANCED_CONTROL_IDS].forEach((id) => {
      setControlAvailability($(`#${id}`), false, "このComfyUIプロファイルは専用設定を使用します");
    });
    $("#profileSummary").textContent = imageMode
      ? "FL2VA · 50ステップ · 開始画像を使用"
      : `FastH3 · 4-Step · VSA ${$("#vsaKeep").value}% · Fast VAE`;
  }
  updateSubmitState();
}

function updateProfileSummary() {
  const profile = selectedProfile();
  if (profile === "fasth3" || profile === "fasth3_fast") {
    $("#profileSummary").textContent = profile === "fasth3_fast"
      ? "FastH3 v1 Blackwell · sm100a/FA4/compile · 5 points（4-forward） · T2VAのみ"
      : "FastH3 VSA · 5 points（4-forward） · T2VAのみ";
  } else if (profile !== "comfy_fasth3" && profile !== "comfy_fl2va") {
    const parts = [`${$("#steps").value} points`, $("#attention").value === "sageattn" ? "SageAttention" : "SDPA"];
    if ($("#easycache").checked) parts.push("EasyCache");
    if ($("#pdd").checked) parts.push("PDD 8-Step");
    $("#profileSummary").textContent = `${PROFILE_LABELS[profile] || profile} · ${parts.join(" · ")}`;
  }
  const seed = $("#seedRandom").checked ? "Seed ランダム" : `Seed ${$("#seed").value || "—"}`;
  const gpu = $("#gpuDevice").value && $("#gpuDevice").value !== "auto" ? "GPU指定" : "GPU Auto";
  $("#advancedSummary").textContent = `${seed} · ${gpu} · ${$("#task").value === "auto" ? "タスク自動" : $("#task").value}`;
}

function profileCanGenerate(profile = selectedProfile()) {
  if (profile === "comfy_fasth3") return state.modelSetup?.comfy_fasth3?.ready === true;
  if (profile === "comfy_fl2va") return state.modelSetup?.comfy_fl2va?.ready === true;
  const capabilities = modelCapabilities();
  if (profile === "fasth3" || profile === "fasth3_fast") {
    return profile === "fasth3_fast" ? capabilities.fastH3FastReady : capabilities.fastH3Ready;
  }
  if (profile === "pdd" || profile === "pdd_sage") return capabilities.pddReady;
  return capabilities.known && capabilities.coreReady;
}

function anyProfileReady() {
  return $$('input[name="profile"]').some((radio) => !radio.disabled);
}

function updateSubmitState() {
  const button = $("#generateButton");
  const hint = $("#submitHint");
  if (!button || !hint) return;
  let reason = "";
  if (!state.modelSetupLoaded) reason = "モデル状態を確認中…";
  else if (!profileCanGenerate(selectedProfile())) {
    reason = "選択中のプロファイルに必要なモデルが未準備です。設定 → モデルを準備 から取得してください";
  }
  button.disabled = Boolean(reason) || state.submitting;
  hint.textContent = reason;
  hint.hidden = !reason;
}

function updateSetupBanner() {
  const banner = $("#setupBanner");
  if (!banner) return;
  const capabilities = modelCapabilities();
  const show = state.modelSetupLoaded && !anyProfileReady();
  banner.hidden = !show;
  const badge = $("#settingsBadge");
  if (badge) badge.hidden = !show;
  if (!show) return;
  const comfyMessage = state.modelSetup?.comfy_fasth3?.message;
  $("#setupBannerText").textContent = capabilities.known
    ? `通常H3の未準備: ${modelMissingLabel(capabilities.coreMissing) || "なし"}${comfyMessage ? `。FastH3: ${comfyMessage}` : ""}`
    : "モデル状態を取得できませんでした。設定画面で再スキャンしてください。";
}

function updateEngineStatus() {
  const pill = $("#enginePill");
  const label = $("#engineState");
  if (!pill || !label) return;
  const active = state.jobs.filter(isActive).length;
  const paths = Object.values(state.bootstrap?.readiness || {});
  let mode = "";
  let text = "確認中";
  if (state.connectionLost) [mode, text] = ["error", "接続が切れました"];
  else if (state.bootstrapError) [mode, text] = ["error", "接続エラー"];
  else if (!state.modelSetupLoaded) [mode, text] = ["", "確認中"];
  else if (active) [mode, text] = ["busy", `生成中 · ${active}件`];
  else if (anyProfileReady()) [mode, text] = ["ready", "生成可能"];
  else if (paths.length && !paths.every((item) => item.ready)) [mode, text] = ["warn", "設定を確認"];
  else [mode, text] = ["warn", "モデル未準備"];
  pill.className = `engine-pill ${mode}`.trim();
  label.textContent = text;
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
const FL2VA_MODEL_IDS = ["transformer_w4a8", "text_encoder_nvfp4_awq", "video_vae_int8_convrot", "audio_vae_fp32"];
const MODEL_ASSET_SHORT_LABELS = {
  transformer_w4a8: "DiT",
  text_encoder_nvfp4_awq: "TEXT",
  video_vae_int8_convrot: "VIDEO VAE",
  audio_vae_fp32: "AUDIO VAE",
  checkpoint_support: "support files",
  pdd_fl2va_8step: "PDD LoRA",
  pdd_adaln_affine: "AdaLN",
  transformer_fastvideo_vsa_4step: "FastH3 DiT",
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
  const radio = profileRadio(profile);
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
  const wrapper = control.closest("label") || control.closest(".option-box");
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
  const fl2vaStatus = state.modelSetup?.comfy_fl2va;
  setProfileAvailability("comfy_fl2va", fl2vaStatus?.ready === true, fl2vaStatus?.message || "画像用モデル・実行環境を確認中");
  $("#comfyReadiness").textContent = comfyStatus?.message || "FastH3環境は未確認です。再スキャンしてください";
  $("#comfyReadiness").className = `availability-note ${comfyStatus?.ready ? "ready" : comfyStatus ? "blocked" : ""}`.trim();
  ["fast_sage", "fast_sage_detail", "fast", "quality", "custom"].forEach((profile) => {
    setProfileAvailability(profile, standardEnabled, standardEnabled ? "" : standardReason);
  });
  setProfileAvailability("pdd", capabilities.pddReady, capabilities.pddReady ? "" : capabilities.coreReady ? pddReason : standardReason);
  setProfileAvailability(
    "fasth3",
    capabilities.fastH3Ready,
    capabilities.fastH3Ready ? "" : (state.fastH3Status ? "FastH3の利用条件を満たしていません" : "設定画面で「利用条件を診断」を実行してください"),
  );
  setProfileAvailability(
    "fasth3_fast",
    capabilities.fastH3FastReady,
    capabilities.fastH3FastReady ? "" : "Blackwell向けFastH3の利用条件を満たしていません",
  );
  const fastProfile = profileRadio("fasth3_fast");
  fastProfile?.closest(".profile-card")?.toggleAttribute("hidden", !capabilities.fastH3FastReady);

  if (profileNote) {
    if (!capabilities.known) {
      profileNote.hidden = true;
    } else {
      profileNote.hidden = false;
      profileNote.className = `availability-note ${capabilities.coreReady ? "ready" : "blocked"}`;
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
      advancedNote.className = "availability-note blocked";
      advancedNote.textContent = `モデル依存の詳細設定を無効化しています。${standardReason}`;
    }
  }

  // Return to the operator's explicit choice once it becomes available again.
  const preferred = state.preferredProfile && profileRadio(state.preferredProfile);
  if (preferred && !preferred.disabled && !preferred.checked) {
    checkProfile(preferred);
    applyProfile(preferred.value);
  }
  const checked = $('input[name="profile"]:checked');
  if (!checked || checked.disabled) {
    const fallback = ["comfy_fl2va", "comfy_fasth3", "fast_sage_detail", "fast_sage", "fast", "quality", "pdd", "fasth3", "fasth3_fast", "custom"]
      .map((profile) => profileRadio(profile))
      .find((radio) => radio && !radio.disabled);
    if (fallback) {
      checkProfile(fallback);
      applyProfile(fallback.value);
    }
  }
  updateProfileSummary();
  updateComfyControls();
  updateSetupBanner();
  updateEngineStatus();
}

/* ------------------------------------------------------------------ */
/* Composer: duration, resolution, seed, images                        */
/* ------------------------------------------------------------------ */

function updateDuration() {
  const count = framesFor(state.duration);
  const exact = count / 24;
  $("#frameHint").textContent = `${count} frames`;
  $("#summaryFrames").textContent = `${count} frames`;
  $("#summaryDuration").textContent = `${exact.toFixed(1)}秒`;
  updateOutputSummary();
}

function setDuration(seconds, { fromCustom = false } = {}) {
  const value = Math.min(30, Math.max(1, Number(seconds) || 5));
  state.duration = value;
  const buttons = $$("#durationControl button");
  const preset = buttons.find((button) => Number(button.dataset.duration) === value);
  buttons.forEach((button) => {
    const on = button === preset;
    button.classList.toggle("active", on);
    button.setAttribute("aria-pressed", String(on));
  });
  const input = $("#customDuration");
  input.closest(".segmented-input")?.classList.toggle("active", !preset);
  if (!fromCustom) input.value = preset ? "" : String(value);
  input.removeAttribute("aria-invalid");
  updateDuration();
}

function resolution() {
  const preset = $("#resolutionPreset").value;
  if (preset === "custom") return [Number($("#width").value), Number($("#height").value)];
  return preset.split("x").map(Number);
}

function snapDimension(value) {
  return Math.min(1536, Math.max(256, Math.round((Number(value) || 512) / 32) * 32));
}

function setResolution(width, height) {
  const key = `${width}x${height}`;
  if (RESOLUTION_PRESETS[key] && key !== "custom") {
    $("#resolutionPreset").value = key;
  } else {
    $("#resolutionPreset").value = "custom";
    $("#width").value = snapDimension(width);
    $("#height").value = snapDimension(height);
  }
  updateResolution();
}

function updateResolution() {
  const preset = $("#resolutionPreset").value;
  const custom = preset === "custom";
  $("#customResolution").hidden = !custom;
  const [width, height] = resolution();
  ["#width", "#height"].forEach((selector) => {
    const value = Number($(selector).value);
    const valid = Number.isFinite(value) && value >= 256 && value <= 1536 && value % 32 === 0;
    if (custom && !valid) $(selector).setAttribute("aria-invalid", "true");
    else $(selector).removeAttribute("aria-invalid");
  });
  $("#resolutionHint").textContent = `${width} × ${height}`;
  const info = RESOLUTION_PRESETS[preset] || RESOLUTION_PRESETS.custom;
  const tier = $("#resolutionTier");
  tier.textContent = info.tier;
  tier.className = `tier-chip ${info.className}`;
  $("#resolutionDescription").textContent = custom
    ? `${info.description}（現在 ${width} × ${height}）`
    : info.description;
  const box = $("#aspectBox");
  if (box && width > 0 && height > 0) {
    const scale = 26 / Math.max(width, height);
    box.style.width = `${Math.max(6, Math.round(width * scale))}px`;
    box.style.height = `${Math.max(6, Math.round(height * scale))}px`;
  }
  updateOutputSummary();
  updateImageTools();
}

function updateOutputSummary() {
  const [width, height] = resolution();
  const text = `${width} × ${height}`;
  const seconds = (framesFor(state.duration) / 24).toFixed(1);
  const summary = $("#outputSummary");
  if (summary) summary.textContent = `${seconds}秒 · ${text}`;
  const resolutionSummary = $("#summaryResolution");
  if (resolutionSummary) resolutionSummary.textContent = text;
}

function updateSeedMode() {
  const random = $("#seedRandom").checked;
  $("#seed").disabled = random;
  $("#seedModeHint").textContent = random ? "生成ごとに自動抽選" : "固定値で再現";
  updateProfileSummary();
}

function updatePromptCount() {
  $("#promptCount").textContent = $("#prompt").value.length.toLocaleString("ja-JP");
}

function updatePromptTransformNote() {
  $("#promptTransformNote").hidden = !state.promptTransform;
}

function setPrompt(value) {
  $("#prompt").value = value;
  updatePromptCount();
}

function setImageAsset(last, asset) {
  const prefix = last ? "lastImage" : "image";
  state[last ? "lastImageAsset" : "imageAsset"] = asset;
  state.imageSizes[last ? "last" : "first"] = null;
  const preview = $(`#${prefix}Preview`);
  if (asset) {
    $(`#${prefix}Name`).textContent = asset.name || "アップロード済み画像";
    $(`#${prefix}Name`).title = asset.name || "";
    preview.src = asset.url;
    preview.hidden = false;
  } else {
    $(`#${prefix}Name`).textContent = "画像未選択";
    $(`#${prefix}Name`).title = "";
    $(`#${prefix}File`).value = "";
    preview.removeAttribute("src");
    preview.hidden = true;
  }
  $(last ? "#removeLastImage" : "#removeImage").hidden = !asset;
  updateImageTools();
}

async function uploadImage(file, last = false) {
  if (!file) return;
  if (!IMAGE_TYPES.includes(file.type)) {
    toast("PNG・JPEG・WebPの画像を選択してください", "error");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    toast(`画像が大きすぎます（${bytes(file.size)}）。25 MB以下にしてください`, "error");
    return;
  }
  const form = new FormData();
  // Pasted images have a generic name; keep a valid extension for the server check.
  const extension = { "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp" }[file.type];
  const name = /\.(png|jpe?g|webp)$/i.test(file.name || "") ? file.name : `pasted-image${extension}`;
  form.append("file", file, name);
  const dropzone = $(last ? "#lastDropzone" : "#dropzone");
  dropzone.classList.add("loading");
  try {
    const asset = await api("/api/assets", { method: "POST", body: form });
    setImageAsset(last, { ...asset, name });
    const task = $("#task");
    if (!task.disabled && !["comfy_fasth3", "comfy_fl2va"].includes(selectedProfile())) task.value = "auto";
    updateProfileSummary();
    saveDraft();
    toast(`${last ? "終了" : "開始"}画像を読み込みました`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    dropzone.classList.remove("loading");
  }
}

function removeImage(last = false) {
  setImageAsset(last, null);
  saveDraft();
}

function restoreAsset(assetId, last, name) {
  if (!assetId) {
    setImageAsset(last, null);
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    const probe = new Image();
    probe.onload = () => {
      setImageAsset(last, { id: assetId, url: `/api/assets/${assetId}`, name: name || "前回の画像" });
      resolve(true);
    };
    probe.onerror = () => {
      setImageAsset(last, null);
      resolve(false);
    };
    probe.src = `/api/assets/${assetId}`;
  });
}

function swapImages() {
  const first = state.imageAsset;
  const last = state.lastImageAsset;
  setImageAsset(false, last);
  setImageAsset(true, first);
  saveDraft();
}

function bestResolutionFor(ratio) {
  const [currentWidth, currentHeight] = resolution();
  const area = (currentWidth || 512) * (currentHeight || 512);
  const presets = Object.keys(RESOLUTION_PRESETS)
    .filter((key) => key !== "custom")
    .map((key) => key.split("x").map(Number));
  const close = presets.filter(([w, h]) => Math.abs(Math.log((w / h) / ratio)) < 0.04);
  if (close.length) {
    return close.sort((a, b) => Math.abs(a[0] * a[1] - area) - Math.abs(b[0] * b[1] - area))[0];
  }
  const height = Math.sqrt(area / ratio);
  return [snapDimension(height * ratio), snapDimension(height)];
}

function updateImageTools() {
  const tools = $("#imageTools");
  if (!tools) return;
  const hasAny = Boolean(state.imageAsset || state.lastImageAsset);
  tools.hidden = !hasAny;
  if (!hasAny) return;
  $("#swapImages").disabled = !(state.imageAsset && state.lastImageAsset);
  const note = $("#imageAspectNote");
  const size = state.imageSizes.first;
  const match = $("#matchImageResolution");
  if (state.lastImageAsset && !state.imageAsset) {
    note.textContent = "終了画像だけでは生成できません。開始画像も選択してください。";
    note.className = "image-aspect-note warn";
    match.hidden = true;
    return;
  }
  if (!size) {
    note.textContent = "";
    note.className = "image-aspect-note";
    match.hidden = true;
    return;
  }
  const [width, height] = resolution();
  const imageRatio = size.width / size.height;
  const mismatch = width > 0 && height > 0 && Math.abs(Math.log((width / height) / imageRatio)) > 0.04;
  note.textContent = mismatch
    ? `開始画像 ${size.width} × ${size.height} と出力 ${width} × ${height} の縦横比が異なります。画像は切り抜き・変形される場合があります。`
    : `開始画像 ${size.width} × ${size.height} · 出力の縦横比と一致しています`;
  note.className = `image-aspect-note${mismatch ? " warn" : ""}`;
  match.hidden = !mismatch;
}

function matchImageResolution() {
  const size = state.imageSizes.first;
  if (!size) return;
  const [width, height] = bestResolutionFor(size.width / size.height);
  setResolution(width, height);
  saveDraft();
  toast(`解像度を ${width} × ${height} に変更しました`, "info");
}

/* ------------------------------------------------------------------ */
/* Draft persistence                                                   */
/* ------------------------------------------------------------------ */

function draftSnapshot() {
  const profile = $('input[name="profile"]:checked')?.value || selectedProfile();
  const advanced = {};
  ADVANCED_CONTROL_IDS.forEach((id) => {
    const control = $(`#${id}`);
    advanced[id] = control.type === "checkbox" ? control.checked : control.value;
  });
  return {
    prompt: $("#prompt").value,
    promptTransform: state.promptTransform,
    duration: state.duration,
    resolution: $("#resolutionPreset").value,
    width: $("#width").value,
    height: $("#height").value,
    seed: $("#seed").value,
    seedRandom: $("#seedRandom").checked,
    profile,
    vsaKeep: $("#vsaKeep").value,
    fastVaeBatch: $("#fastVaeBatch").value,
    gpuDevice: $("#gpuDevice").value,
    promptCache: $("#promptCache").checked,
    advanced: profile === "custom" ? advanced : null,
    image: state.imageAsset ? { id: state.imageAsset.id, name: state.imageAsset.name } : null,
    lastImage: state.lastImageAsset ? { id: state.lastImageAsset.id, name: state.lastImageAsset.name } : null,
  };
}

function saveDraft() {
  clearTimeout(state.draftTimer);
  state.draftTimer = setTimeout(() => {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(draftSnapshot())); } catch { /* optional */ }
  }, 300);
}

function restoreDraft() {
  let draft = null;
  try { draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null"); } catch { draft = null; }
  if (!draft || typeof draft !== "object") return;
  if (typeof draft.prompt === "string") setPrompt(draft.prompt.slice(0, 12000));
  if (draft.promptTransform?.original) state.promptTransform = draft.promptTransform;
  setDuration(draft.duration || 5);
  if (draft.resolution && $(`#resolutionPreset option[value="${CSS.escape(draft.resolution)}"]`)) {
    $("#resolutionPreset").value = draft.resolution;
    if (draft.resolution === "custom") {
      $("#width").value = snapDimension(draft.width);
      $("#height").value = snapDimension(draft.height);
    }
  }
  if (draft.seed !== undefined && draft.seed !== "") $("#seed").value = draft.seed;
  $("#seedRandom").checked = Boolean(draft.seedRandom);
  if (draft.vsaKeep) $("#vsaKeep").value = draft.vsaKeep;
  if (draft.fastVaeBatch) $("#fastVaeBatch").value = draft.fastVaeBatch;
  if (typeof draft.promptCache === "boolean") $("#promptCache").checked = draft.promptCache;
  if (draft.gpuDevice) state.draftGpuDevice = draft.gpuDevice;
  const radio = draft.profile && profileRadio(draft.profile);
  if (radio) {
    state.preferredProfile = draft.profile;
    checkProfile(radio);
  }
  if (draft.profile === "custom" && draft.advanced) {
    ADVANCED_CONTROL_IDS.forEach((id) => {
      const control = $(`#${id}`);
      if (!(id in draft.advanced)) return;
      if (control.type === "checkbox") control.checked = Boolean(draft.advanced[id]);
      else control.value = draft.advanced[id];
    });
  }
  restoreAsset(draft.image?.id, false, draft.image?.name);
  restoreAsset(draft.lastImage?.id, true, draft.lastImage?.name);
}

/* ------------------------------------------------------------------ */
/* Payload and submission                                              */
/* ------------------------------------------------------------------ */

function generationPayload() {
  const [width, height] = resolution();
  const randomSeed = Boolean($("#seedRandom").checked);
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
    seed: randomSeed ? null : Number($("#seed").value),
    filename: "hayate",
    image_asset_id: state.imageAsset?.id || null,
    last_image_asset_id: state.lastImageAsset?.id || null,
    reference_asset_ids: [],
    use_prompt_cache: !["comfy_fasth3", "comfy_fl2va"].includes(selectedProfile()) && $("#promptCache").checked,
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

function validatePayload(payload) {
  if (!payload.prompt) return "プロンプトを入力してください";
  if (payload.profile === "comfy_fasth3" && (payload.image_asset_id || payload.last_image_asset_id)) return "FastH3はテキスト専用です。画像優先 FL2VAを選択してください";
  if (payload.profile === "comfy_fl2va" && !payload.image_asset_id) return "画像優先 FL2VAには開始画像が必要です";
  if (payload.last_image_asset_id && !payload.image_asset_id) return "終了画像を使う場合は開始画像も選択してください";
  const dimensions = [payload.width, payload.height];
  if (dimensions.some((value) => !Number.isFinite(value) || value < 256 || value > 1536 || value % 32)) {
    return "解像度は幅・高さとも256〜1536の32の倍数で指定してください";
  }
  if (!(payload.duration_seconds >= 1 && payload.duration_seconds <= 30)) return "動画の長さは1〜30秒で指定してください";
  if (payload.seed !== null && !(Number.isInteger(payload.seed) && payload.seed >= 0 && payload.seed <= 2147483647)) {
    return "Seedは0〜2147483647の整数で指定してください";
  }
  return "";
}

async function submitGeneration(event) {
  event?.preventDefault();
  if (state.submitting) return;
  const payload = generationPayload();
  const problem = validatePayload(payload);
  if (problem) {
    toast(problem, "error");
    if (!payload.prompt) $("#prompt").focus();
    return;
  }
  if (payload.profile === "comfy_fl2va" && state.imageSizes.first) {
    const { width, height } = state.imageSizes.first;
    if (Math.abs(Math.log((payload.width / payload.height) / (width / height))) > 0.04) {
      toast("開始画像と出力の縦横比が異なります。『画像に合わせる』を押してください", "error", {
        action: { label: "画像に合わせる", run: matchImageResolution },
      });
      return;
    }
  }
  if (!profileCanGenerate(payload.profile)) {
    toast("選択した構成のモデルが未取得または未検証です。設定画面でモデル状態を確認してください", "error", {
      action: { label: "設定を開く", run: () => navigate("settings") },
    });
    return;
  }
  const button = $("#generateButton");
  state.submitting = true;
  button.classList.add("busy");
  button.disabled = true;
  $("#generateLabel").textContent = "プリフライト中…";
  try {
    const job = await api("/api/jobs", { method: "POST", body: payload });
    upsertJob(job);
    state.activeJobId = job.id;
    renderLive(job);
    connectEvents(job.id);
    renderJobs();
    const seed = job.request?.seed;
    toast(`生成キューへ追加しました${Number.isFinite(Number(seed)) ? `（Seed ${seed}）` : ""}`);
    if (window.matchMedia("(max-width: 980px)").matches) $("#liveCard").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.submitting = false;
    button.classList.remove("busy");
    $("#generateLabel").textContent = "生成を開始";
    updateSubmitState();
  }
}

/* ------------------------------------------------------------------ */
/* Prompt assistant                                                    */
/* ------------------------------------------------------------------ */

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
  if (!brief) {
    $("#promptBrief").focus();
    return setPromptAIStatus("映像の概要を入力してください", "error");
  }
  const [width, height] = resolution();
  const button = $("#generateAIPrompt");
  button.disabled = true;
  button.classList.add("busy");
  setPromptAIStatus("OpenAIでH3向け構成を作成中…（数十秒かかる場合があります）");
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
  } catch (error) {
    setPromptAIStatus(error.message, "error");
  } finally {
    button.disabled = false;
    button.classList.remove("busy");
  }
}

function openPromptAssist() {
  const original = state.promptTransform?.original || $("#prompt").value.trim();
  state.promptAI = null;
  $("#promptBrief").value = original;
  $("#promptTask").value = $("#task").value;
  $("#promptIncludeAudio").checked = true;
  $("#promptAIResult").hidden = true;
  const configured = state.bootstrap?.openai?.api_key_configured;
  setPromptAIStatus(configured === false
    ? "OpenAI APIキーが未設定です。設定 → AIプロンプト作成 で登録するか、下の3項目を手入力してください。"
    : "");
  $("#promptVisual").value = displayPrompt(original);
  $("#promptSound").value = "";
  $("#promptMusic").value = "";
  structuredPromptDraft();
  state.dialogTrigger = document.activeElement;
  $("#promptDialog").showModal();
  $("#promptBrief").focus();
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
  setPrompt(draft);
  updatePromptTransformNote();
  saveDraft();
  $("#promptDialog").close();
  toast("確認したH3向けプロンプト案を適用しました");
}

/* ------------------------------------------------------------------ */
/* Jobs: live stream, queue, library                                   */
/* ------------------------------------------------------------------ */

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
  source.onmessage = (event) => {
    let job;
    try { job = JSON.parse(event.data); } catch { return; }
    const previous = findJob(job.id);
    upsertJob(job);
    if (state.activeJobId === job.id) renderLive(job);
    renderJobs();
    if (FINAL_STATUSES.includes(job.status)) {
      source.close();
      state.eventSources.delete(job.id);
      if (previous && FINAL_STATUSES.includes(previous.status)) return;
      announceFinished(job);
      refreshJobs();
    }
  };
  source.onerror = () => {
    source.close();
    state.eventSources.delete(jobId);
    setTimeout(() => refreshJobs(), 1500);
  };
}

function announceFinished(job) {
  if (["succeeded", "partial"].includes(job.status)) {
    toast(job.status === "partial" ? "途中状態を保存しました" : "動画が完成しました", "success", {
      action: { label: "見る", run: () => openJob(job.id) },
    });
    if (document.hidden) document.title = "✓ 完成 · HAYATE Studio";
  } else if (job.status === "failed") {
    toast(job.error || "生成に失敗しました", "error", { action: { label: "ログ", run: () => openJob(job.id) } });
    if (document.hidden) document.title = "× 失敗 · HAYATE Studio";
  }
}

function activeElapsed(job) {
  if (!job?.started_at) return 0;
  const end = job.completed_at ? new Date(job.completed_at) : new Date();
  return Math.max(0, (end - new Date(job.started_at)) / 1000);
}

function updateDocumentTitle(job) {
  if (document.hidden && /^[✓×]/.test(document.title)) return;
  if (job && ["running", "stopping", "cancelling"].includes(job.status)) {
    document.title = `${Math.round(Number(job.progress || 0))}% 生成中 · HAYATE Studio`;
  } else if (job?.status === "queued") {
    document.title = "待機中 · HAYATE Studio";
  } else {
    document.title = "HAYATE Studio";
  }
}

function renderLive(job) {
  const idle = !job;
  $("#idleState").hidden = !idle;
  $("#progressState").hidden = idle;
  $("#liveCard").classList.toggle("running", isActive(job));
  $(".generate-grid")?.classList.toggle("live-first", !idle);
  updateDocumentTitle(job);
  if (idle) {
    $("#liveStatus").className = "status-chip idle";
    $("#liveStatus").textContent = "待機中";
    return;
  }
  const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
  const final = FINAL_STATUSES.includes(job.status);
  $("#liveStatus").className = `status-chip ${job.status}`;
  $("#liveStatus").textContent = statusLabels[job.status] || job.status;
  $("#liveStage").textContent = job.stage || "準備中";
  $("#liveDetail").textContent = job.detail || "—";
  $("#livePercent").textContent = `${Math.round(progress)}%`;
  $("#liveProgressBar").value = progress;
  $("#liveProgressTrack").setAttribute("aria-valuenow", String(Math.round(progress)));
  $("#elapsedTime").textContent = clock(activeElapsed(job));
  $("#etaTime").textContent = final ? "—" : job.eta_seconds === 0 ? "まもなく" : clock(job.eta_seconds);
  $("#liveProfile").textContent = profileLabel(job);
  $$("#stageList span").forEach((span) => span.classList.toggle("done", progress >= Number(span.dataset.threshold)));

  const failed = ["failed", "cancelled", "interrupted"].includes(job.status);
  $("#liveError").hidden = !failed;
  if (failed) $("#liveError").textContent = job.error || job.detail || statusLabels[job.status];
  $("#liveStopActions").hidden = final;
  $("#liveResultActions").hidden = !final;
  $("#liveOpenButton").textContent = ["succeeded", "partial"].includes(job.status) ? "拡大再生・詳細" : "詳細・ログを見る";
  const runnable = ["queued", "running"].includes(job.status);
  $("#stopSaveButton").disabled = !runnable || job.plan?.backend === "comfy_fasth3";
  $("#stopSaveButton").title = job.plan?.backend === "comfy_fasth3" ? "FastH3 ComfyUIは途中保存に未対応です" : "";
  $("#cancelButton").disabled = !runnable && job.status !== "stopping";
  $("#previewLabel").textContent = ["succeeded", "partial"].includes(job.status)
    ? "COMPLETE"
    : String(job.stage || "GENERATING").toUpperCase();
  const preview = $("#livePreview");
  let video = $("video", preview);
  const playable = ["succeeded", "partial"].includes(job.status) && job.media_available !== false;
  if (playable) {
    if (!video) {
      video = document.createElement("video");
      video.muted = true; video.loop = true; video.autoplay = true; video.playsInline = true;
      video.title = "クリックで拡大再生";
      video.addEventListener("click", () => openJob(state.activeJobId));
      preview.prepend(video);
    }
    const src = `/api/jobs/${job.id}/media`;
    if (video.getAttribute("src") !== src) video.src = src;
  } else if (video) {
    video.remove();
  }
}

async function refreshJobs() {
  try {
    const payload = await api("/api/jobs?limit=200");
    state.jobs = payload.jobs;
    renderJobs();
    const activeJobs = state.jobs.filter(isActive);
    activeJobs.forEach((job) => {
      if (!state.eventSources.has(job.id)) connectEvents(job.id);
    });
    for (const [jobId, source] of state.eventSources) {
      if (!activeJobs.some((job) => job.id === jobId)) {
        source.close();
        state.eventSources.delete(jobId);
      }
    }
    const current = findJob(state.activeJobId);
    if (!isActive(current)) {
      // Prefer the job that is actually running over one still waiting.
      const live = activeJobs.find((job) => job.status !== "queued") || activeJobs[activeJobs.length - 1];
      if (live) state.activeJobId = live.id;
    }
    renderLive(findJob(state.activeJobId));
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderJobs() {
  const active = state.jobs.filter(isActive);
  $("#queueBadge").hidden = !active.length;
  $("#queueBadge").textContent = active.length;
  renderQueue(active);
  renderRecent();
  renderLibrary();
  updateEngineStatus();
}

function renderQueue(active) {
  $("#queueActiveCount").textContent = `${active.length}件`;
  const ordered = [...active].sort((a, b) => {
    const rank = (job) => (job.status === "queued" ? 1 : 0);
    return rank(a) - rank(b) || new Date(a.created_at) - new Date(b.created_at);
  });
  $("#queueActive").innerHTML = ordered.length
    ? ordered.map((job, index) => activeJobRow(job, index)).join("")
    : '<div class="empty-list">実行中のジョブはありません。生成画面から新しいジョブを追加できます。</div>';
  const history = state.jobs.filter((job) => job.source === "webui" && !isActive(job)).slice(0, 30);
  $("#queueList").innerHTML = history.length
    ? history.map((job, index) => historyJobRow(job, index)).join("")
    : '<div class="empty-list">まだWebUIから生成したジョブはありません</div>';
}

function activeJobRow(job, index) {
  const prompt = displayPrompt(jobPrompt(job)) || "生成ジョブ";
  const progress = Number(job.progress || 0);
  const eta = job.status === "queued" ? "待機中" : `${Math.round(progress)}% · 残り ${clock(job.eta_seconds)}`;
  const comfy = job.plan?.backend === "comfy_fasth3";
  const canStop = job.status === "running" && !comfy;
  const cancelLabel = job.status === "queued" ? "取り消す" : "中止";
  return `<article class="job-row" data-job="${job.id}">
    <div class="job-index">${String(index + 1).padStart(2, "0")}</div>
    <div class="job-main"><b title="${escapeHTML(prompt)}">${escapeHTML(prompt)}</b><small>${escapeHTML(job.stage || "")} · ${escapeHTML(job.detail || "")}</small><div class="mini-progress"><progress max="100" value="${progress}"></progress></div></div>
    <div class="job-stat"><span>状態</span><b><span class="status-chip ${job.status}">${statusLabels[job.status] || job.status}</span></b></div>
    <div class="job-stat"><span>進捗</span><b>${escapeHTML(eta)}</b></div>
    <div class="job-stat gpu"><span>GPU · プロファイル</span><b title="${escapeHTML(gpuLabel(job))}">${escapeHTML(gpuLabel(job))}</b></div>
    <div class="job-actions">
      <button type="button" class="secondary-button" data-live-job="${job.id}">表示</button>
      ${canStop ? `<button type="button" class="warn-button" data-stop-job="${job.id}">途中保存</button>` : ""}
      ${["queued", "running", "stopping"].includes(job.status) ? `<button type="button" class="danger-button" data-cancel-job="${job.id}">${cancelLabel}</button>` : ""}
    </div>
  </article>`;
}

function historyJobRow(job, index) {
  const prompt = displayPrompt(jobPrompt(job)) || "生成結果";
  const request = jobRequest(job);
  const size = request.width && request.height ? `${request.width}×${request.height}` : "";
  return `<article class="job-row" data-job="${job.id}">
    <div class="job-index">${String(index + 1).padStart(2, "0")}</div>
    <div class="job-main"><b title="${escapeHTML(prompt)}">${escapeHTML(prompt)}</b><small>${compactDate(job.created_at)} · ${escapeHTML(profileLabel(job))}${size ? ` · ${size}` : ""}</small></div>
    <div class="job-stat"><span>状態</span><b><span class="status-chip ${job.status}">${statusLabels[job.status] || job.status}</span></b></div>
    <div class="job-stat"><span>生成時間</span><b>${job.duration_seconds != null ? clock(job.duration_seconds) : "—"}</b></div>
    <div class="job-stat gpu"><span>GPU</span><b title="${escapeHTML(gpuLabel(job))}">${escapeHTML(gpuLabel(job))}</b></div>
    <div class="job-actions">
      <button type="button" class="secondary-button" data-open-job="${job.id}">${["succeeded", "partial"].includes(job.status) ? "見る" : "詳細"}</button>
      <button type="button" class="secondary-button" data-reuse-job="${job.id}">再利用</button>
    </div>
  </article>`;
}

function renderRecent() {
  const recent = state.jobs
    .filter((job) => ["succeeded", "partial"].includes(job.status) && job.media_available !== false)
    .slice(0, 4);
  const key = recent.map((job) => job.id).join(",");
  if (key === state.recentKey) return;
  state.recentKey = key;
  $("#recentCard").hidden = !recent.length;
  $("#recentList").innerHTML = recent.map((job) => {
    const prompt = displayPrompt(jobPrompt(job)) || jobFilename(job);
    return `<button type="button" class="recent-item" data-open-job="${job.id}" title="${escapeHTML(prompt)}">
      <video data-src="/api/jobs/${job.id}/media#t=0.5" muted preload="none" playsinline></video>
      <span>${escapeHTML(compactDate(job.created_at))}</span>
    </button>`;
  }).join("");
  observeVideos($("#recentList"));
}

function libraryJobs() {
  const query = ($("#librarySearch")?.value || "").trim().toLowerCase();
  const filter = $("#libraryFilter")?.value || "all";
  const sort = $("#librarySort")?.value || "newest";
  const jobs = state.jobs.filter((job) => {
    if (!["succeeded", "partial", "failed"].includes(job.status)) return false;
    if (filter !== "all" && job.status !== filter) return false;
    if (!query) return true;
    const haystack = `${jobPrompt(job)} ${jobRequest(job).original_prompt || ""} ${job.output_path || ""}`.toLowerCase();
    return haystack.includes(query);
  });
  const area = (job) => (Number(jobRequest(job).width) || 0) * (Number(jobRequest(job).height) || 0);
  const sorters = {
    newest: (a, b) => new Date(b.created_at) - new Date(a.created_at),
    oldest: (a, b) => new Date(a.created_at) - new Date(b.created_at),
    longest: (a, b) => (Number(b.duration_seconds) || 0) - (Number(a.duration_seconds) || 0),
    largest: (a, b) => area(b) - area(a) || new Date(b.created_at) - new Date(a.created_at),
  };
  return jobs.sort(sorters[sort] || sorters.newest);
}

function renderLibrary() {
  const grid = $("#libraryGrid");
  const jobs = libraryJobs();
  const key = [
    $("#librarySearch")?.value, $("#libraryFilter")?.value, $("#librarySort")?.value,
    ...jobs.map((job) => `${job.id}:${job.status}:${job.media_available}`),
  ].join("|");
  if (key === state.libraryKey) return;
  state.libraryKey = key;
  const total = state.jobs.filter((job) => ["succeeded", "partial", "failed"].includes(job.status)).length;
  $("#libraryCount").textContent = jobs.length === total ? `${total}件` : `${jobs.length} / ${total}件`;
  if (!jobs.length) {
    grid.innerHTML = total
      ? '<div class="empty-library"><div><b>条件に一致する生成履歴がありません</b>検索語や絞り込みを変更してください。</div></div>'
      : '<div class="empty-library"><div><b>まだ生成履歴がありません</b>生成画面で最初の映像をつくりましょう。</div></div>';
    return;
  }
  grid.innerHTML = jobs.map((job) => {
    const request = jobRequest(job);
    const success = ["succeeded", "partial"].includes(job.status);
    const playable = success && job.media_available !== false;
    const fullPrompt = jobPrompt(job) || "過去の生成結果";
    const prompt = displayPrompt(fullPrompt);
    const seconds = request.frames ? `${(request.frames / 24).toFixed(1)}s` : request.duration_seconds ? `${request.duration_seconds}s` : "";
    const thumb = playable
      ? `<video data-src="/api/jobs/${job.id}/media#t=0.1" muted preload="none" playsinline loop></video>${seconds ? `<span class="thumb-badge">${seconds}</span>` : ""}<span class="thumb-play" aria-hidden="true">▶</span>`
      : `<div class="thumb-empty"><span aria-hidden="true">!</span>${success ? "動画ファイルが見つかりません" : escapeHTML((job.error || "生成に失敗しました").slice(0, 80))}</div>`;
    const specs = [
      request.width && request.height ? `${request.width}×${request.height}` : "",
      request.frames ? `${request.frames} frames` : "",
      profileLabel(job),
      job.duration_seconds != null ? `⏱ ${clock(job.duration_seconds)}` : "",
      Number.isFinite(Number(request.seed)) && request.seed !== null ? `Seed ${request.seed}` : "",
    ].filter(Boolean);
    return `<article class="library-card ${job.status}" data-job="${job.id}">
      <div class="thumb" data-open-job="${job.id}">${thumb}</div>
      <div class="card-body">
        <div class="card-meta"><span class="status-chip ${job.status}">${statusLabels[job.status]}</span><time datetime="${escapeHTML(job.created_at || "")}">${compactDate(job.created_at)}</time></div>
        <h3 title="${escapeHTML(fullPrompt)}">${escapeHTML(prompt)}</h3>
        <div class="card-specs">${specs.map((spec) => `<span>${escapeHTML(spec)}</span>`).join("")}</div>
        <div class="card-actions">
          <button type="button" class="secondary-button" data-open-job="${job.id}">${playable ? "再生・詳細" : "詳細・ログ"}</button>
          <button type="button" class="secondary-button" data-reuse-job="${job.id}" ${request.prompt ? "" : "disabled"}>再利用</button>
          <button type="button" class="danger-button" data-delete-job="${job.id}" aria-label="この生成を削除">削除</button>
        </div>
      </div>
    </article>`;
  }).join("");
  observeVideos(grid);
}

function loadVideo(video) {
  if (!video?.dataset.src) return;
  video.preload = "metadata";
  video.src = video.dataset.src;
  delete video.dataset.src;
}

// Library thumbnails load only when scrolled into view; a large history no
// longer fetches every MP4's metadata on each render.
function observeVideos(root) {
  const videos = $$("video[data-src]", root);
  if (!("IntersectionObserver" in window)) {
    videos.forEach(loadVideo);
    return;
  }
  state.videoObserver ||= new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      loadVideo(entry.target);
      state.videoObserver.unobserve(entry.target);
    });
  }, { rootMargin: "240px" });
  videos.forEach((video) => state.videoObserver.observe(video));
}

/* ------------------------------------------------------------------ */
/* Job detail dialog, reuse, delete, stop                              */
/* ------------------------------------------------------------------ */

function specItem(label, value) {
  return `<div><dt>${escapeHTML(label)}</dt><dd title="${escapeHTML(value)}">${escapeHTML(value)}</dd></div>`;
}

function openJob(jobId) {
  const job = findJob(jobId);
  if (!job) return;
  if (isActive(job)) {
    state.activeJobId = job.id;
    renderLive(job);
    navigate("generate");
    return;
  }
  const request = jobRequest(job);
  const dialog = $("#videoDialog");
  const playable = ["succeeded", "partial"].includes(job.status) && job.media_available !== false;
  state.dialogJobId = job.id;
  if (!dialog.open) state.dialogTrigger = document.activeElement;
  const video = $("#dialogVideo");
  video.hidden = !playable;
  if (playable) video.src = `/api/jobs/${job.id}/media`;
  else { video.pause(); video.removeAttribute("src"); video.load(); }
  $("#dialogNoMedia").hidden = playable;
  $("#dialogNoMediaTitle").textContent = ["succeeded", "partial"].includes(job.status) ? "動画ファイルが見つかりません" : `${statusLabels[job.status] || job.status}したジョブ`;
  $("#dialogNoMediaText").textContent = ["succeeded", "partial"].includes(job.status)
    ? "履歴は残っていますが、MP4が移動または削除されています。"
    : "下の実行ログで原因を確認できます。設定を再利用して再実行することもできます。";
  $("#dialogBadge").className = `status-chip ${job.status}`;
  $("#dialogBadge").textContent = statusLabels[job.status] || job.status;
  $("#dialogDate").textContent = compactDate(job.created_at, true);
  $("#dialogTitle").textContent = jobFilename(job) || "Generated video";
  $("#dialogPrompt").textContent = jobPrompt(job) || "過去の生成結果";
  const errorText = job.error || (["failed", "cancelled", "interrupted"].includes(job.status) ? job.detail : "");
  $("#dialogError").hidden = !errorText;
  $("#dialogError").textContent = errorText || "";
  const effective = request.effective_profile || {};
  const peak = job.runtime_metrics?.cuda_peak_allocated_bytes;
  const images = [request.image_asset_id ? "開始" : "", request.last_image_asset_id ? "終了" : ""].filter(Boolean).join("・");
  $("#dialogStats").innerHTML = [
    specItem("解像度", request.width && request.height ? `${request.width} × ${request.height}` : "—"),
    specItem("長さ", request.frames ? `${(request.frames / 24).toFixed(1)}秒 · ${request.frames} frames` : "—"),
    specItem("プロファイル", profileLabel(job)),
    specItem("Seed", request.seed != null ? String(request.seed) : "—"),
    specItem("ステップ", effective.steps ?? request.steps ?? "—"),
    specItem("タスク", `${request.task || "—"}${images ? ` · 画像: ${images}` : ""}`),
    specItem("生成時間", job.duration_seconds != null ? clock(job.duration_seconds) : "—"),
    specItem("VRAMピーク", bytes(peak)),
    specItem("GPU", gpuLabel(job)),
    specItem("取得元", job.source === "history" ? "CLI / 取り込み" : "WebUI"),
  ].join("");
  const download = $("#downloadDialogVideo");
  if (playable) {
    download.href = `/api/jobs/${job.id}/media`;
    download.setAttribute("download", jobFilename(job) || "hayate.mp4");
    download.removeAttribute("aria-disabled");
  } else {
    download.removeAttribute("href");
    download.setAttribute("aria-disabled", "true");
  }
  $("#reuseDialogJob").disabled = !request.prompt;
  const logBox = $("#dialogLogBox");
  const log = $("#dialogLog");
  log.textContent = "読み込み中…";
  delete log.dataset.loaded;
  logBox.open = job.status === "failed";
  if (!dialog.open) dialog.showModal();
  if (logBox.open) loadDialogLog();
}

async function loadDialogLog() {
  const jobId = state.dialogJobId;
  const log = $("#dialogLog");
  if (!jobId || log.dataset.loaded === jobId) return;
  log.dataset.loaded = jobId;
  log.textContent = "読み込み中…";
  try {
    const result = await api(`/api/jobs/${jobId}/log?lines=400`);
    if (state.dialogJobId !== jobId) return;
    log.textContent = result.available && result.lines.length ? result.lines.join("\n") : "ログファイルがありません";
    log.scrollTop = log.scrollHeight;
  } catch (error) {
    delete log.dataset.loaded;
    log.textContent = error.message;
  }
}

function restoreCustomControls(request) {
  const values = {
    steps: request.steps, attention: request.attention_backend, easycache: request.easycache, pdd: request.pdd,
    ecThreshold: request.easycache_threshold, ecStart: request.easycache_start, ecEnd: request.easycache_end,
    ecSkips: request.easycache_max_consecutive_skips, blocksSwap: request.blocks_to_swap,
    chunkRows: request.activation_chunk_rows, vaeTile: request.vae_tile_size,
  };
  Object.entries(values).forEach(([id, value]) => {
    if (value === undefined || value === null) return;
    const control = $(`#${id}`);
    if (control.type === "checkbox") control.checked = Boolean(value);
    else control.value = value;
  });
}

async function reuseJob(jobId) {
  const job = findJob(jobId);
  const request = jobRequest(job);
  if (!job || !request.prompt) {
    toast("再利用できる設定がありません", "error");
    return;
  }
  const effective = request.effective_prompt || request.prompt;
  setPrompt(effective);
  state.promptTransform = request.prompt_transform_applied && request.original_prompt && request.original_prompt !== effective
    ? { original: request.original_prompt, version: request.prompt_transform_template_version || "h3-base-fields-v1" }
    : null;
  updatePromptTransformNote();
  setDuration(Number(request.duration_seconds) || 5);
  if (request.width && request.height) setResolution(Number(request.width), Number(request.height));
  if (request.seed != null && Number.isFinite(Number(request.seed))) {
    $("#seed").value = request.seed;
    $("#seedRandom").checked = false;
    updateSeedMode();
  }
  if (request.vsa_keep) $("#vsaKeep").value = String(request.vsa_keep).replace(/\.0$/, "");
  if (request.fast_vae_batch) $("#fastVaeBatch").value = String(request.fast_vae_batch);
  const radio = request.profile && profileRadio(request.profile);
  let profileNote = "";
  if (radio && !radio.disabled) {
    state.preferredProfile = request.profile;
    checkProfile(radio);
    applyProfile(request.profile);
    if (request.profile === "custom") restoreCustomControls(request);
  } else if (request.profile) {
    profileNote = `（「${PROFILE_LABELS[request.profile] || request.profile}」は現在利用できないため、プロファイルは変更していません）`;
  }
  if (request.task && !$("#task").disabled) $("#task").value = request.task;
  const [firstOk, lastOk] = await Promise.all([
    restoreAsset(request.image_asset_id, false),
    restoreAsset(request.last_image_asset_id, true),
  ]);
  updateProfileSummary();
  if ($("#videoDialog").open) $("#videoDialog").close();
  navigate("generate");
  saveDraft();
  const missingImage = !firstOk || !lastOk ? "。元の画像は見つからなかったため外しました" : "";
  toast(`生成設定を読み込みました${profileNote}${missingImage}`, profileNote || missingImage ? "info" : "success");
}

function confirmAction({ title, body = "", detailTitle = "", detail = "", confirmLabel = "実行", danger = false, eyebrow = "" }) {
  const dialog = $("#confirmDialog");
  return new Promise((resolve) => {
    dialog.classList.toggle("danger", danger);
    $("#confirmEyebrow").textContent = eyebrow || (danger ? "CONFIRM" : "CONFIRM");
    $("#confirmIcon").textContent = danger ? "×" : "!";
    $("#confirmDialogTitle").textContent = title;
    $("#confirmDialogBody").textContent = body;
    const box = $("#confirmDialogDetail");
    box.replaceChildren();
    if (detailTitle) {
      const heading = document.createElement("b");
      heading.textContent = detailTitle;
      box.append(heading);
    }
    if (detail) box.append(document.createTextNode(detail));
    box.hidden = !detailTitle && !detail;
    $("#confirmAccept").textContent = confirmLabel;
    const trigger = document.activeElement;
    dialog.returnValue = "";
    dialog.addEventListener("close", () => {
      resolve(dialog.returnValue === "confirm");
      if (trigger?.isConnected) trigger.focus?.();
    }, { once: true });
    dialog.showModal();
    $("#confirmCancel").focus();
  });
}

async function deleteJob(jobId) {
  const job = findJob(jobId);
  if (!job) return toast("削除する履歴が見つかりません", "error");
  if (!FINAL_STATUSES.includes(job.status)) return toast("実行中または待機中の生成は削除できません", "error");
  const ok = await confirmAction({
    danger: true,
    eyebrow: "DELETE GENERATION",
    title: "この生成を削除しますか？",
    body: displayPrompt(jobPrompt(job)).slice(0, 220) || "過去の生成結果",
    detailTitle: jobFilename(job) || "履歴のみ",
    detail: "生成履歴・動画・ログ・実行マニフェストを完全に削除します。この操作は元に戻せません。",
    confirmLabel: "完全に削除",
  });
  if (!ok) return;
  try {
    const result = await api(`/api/jobs/${jobId}`, { method: "DELETE" });
    state.jobs = state.jobs.filter((item) => item.id !== jobId);
    if (state.activeJobId === jobId) {
      state.activeJobId = null;
      renderLive(null);
    }
    if (state.dialogJobId === jobId && $("#videoDialog").open) $("#videoDialog").close();
    renderJobs();
    const count = result.artifacts_deleted?.length || 0;
    toast(count ? `生成履歴と関連ファイル${count}件を削除しました` : "生成履歴を削除しました");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function stopJob(jobId, save) {
  const job = findJob(jobId);
  if (!job) return;
  if (!save && job.status !== "queued") {
    const ok = await confirmAction({
      danger: true,
      eyebrow: "CANCEL GENERATION",
      title: "生成を今すぐ中止しますか？",
      body: "生成プロセスを停止します。現在の映像は保存されない場合があります。\n途中までの結果を残す場合は「途中で停止して保存」を使ってください。",
      confirmLabel: "今すぐ中止",
    });
    if (!ok) return;
  }
  try {
    const updated = await api(`/api/jobs/${jobId}/${save ? "stop-and-save" : "cancel"}`, { method: "POST" });
    upsertJob(updated);
    if (state.activeJobId === jobId) renderLive(updated);
    renderJobs();
    toast(save ? "次のステップで停止して保存します" : job.status === "queued" ? "待機中のジョブを取り消しました" : "停止要求を送りました", "info");
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Settings, hardware, model setup                                     */
/* ------------------------------------------------------------------ */

function markSettingsDirty(dirty) {
  state.settingsDirty = dirty;
  $("#settingsDirty").hidden = !dirty;
  $("#saveSettings").classList.toggle("dirty", dirty);
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
  let missing = 0;
  $$("[data-ready]").forEach((dot) => {
    const ready = Boolean(readiness?.[dot.dataset.ready]?.ready);
    if (!ready) missing += 1;
    dot.classList.toggle("ready", ready);
    dot.title = ready ? "見つかりました" : (readiness?.[dot.dataset.ready]?.reason || "見つかりません");
  });
  $("#pathSettingsSummary").textContent = missing ? `${missing}件のパスが見つかりません` : "すべてのパスを確認済み";
  if (state.bootstrap) state.bootstrap.readiness = readiness;
  markSettingsDirty(false);
  updateEngineStatus();
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
  const html = options.map((option) => `<option value="${escapeHTML(option.value)}"${option.disabled ? " disabled" : ""}>${escapeHTML(option.label)}</option>`).join("");
  const valid = (value) => options.some((option) => option.value === value && !option.disabled);
  // The composer keeps the operator's own choice; only Settings follows the saved default.
  const generation = $("#gpuDevice");
  const wanted = state.draftGpuDevice || generation.value || "auto";
  generation.innerHTML = html;
  generation.value = valid(wanted) ? wanted : "auto";
  state.draftGpuDevice = null;
  const settingsSelect = $("#settingGpuDefault");
  settingsSelect.innerHTML = html;
  settingsSelect.value = valid(defaultSelector) ? defaultSelector : "auto";
}

function renderNetwork(network) {
  const exposed = network?.exposed === true;
  const panel = $("#networkPanel");
  panel.classList.toggle("exposed", exposed);
  panel.classList.toggle("local", !exposed);
  $("#networkMode").textContent = exposed ? "LAN" : "LOCAL";
  $("#networkEyebrow").textContent = exposed ? "LAN OPEN" : "LOCAL ONLY";
  $("#networkTitle").textContent = exposed ? "LANからアクセスできます" : "このPCからのみアクセスできます";
  $("#networkText").textContent = exposed
    ? "認証はありません。公開URLを知る人はOpenAI設定とAIプロンプト作成を利用できます。共有範囲に注意してください。"
    : "ループバック（127.0.0.1）でのみ待ち受けています。LANから使う場合は --host 0.0.0.0 --allow-network で起動してください。";
}

function modelSourceUrls(assetId) {
  return Object.fromEntries(Object.entries(state.modelSourceUrls[assetId] || {})
    .map(([path, url]) => [path, url.trim()]).filter(([, url]) => url));
}

function configurationAssets(assets) {
  const choice = $("#modelConfiguration").value;
  if (choice === "all") return assets;
  if (choice === "comfy") return assets.filter((asset) => COMFY_MODEL_IDS.includes(asset.id));
  if (choice === "fl2va") return assets.filter((asset) => FL2VA_MODEL_IDS.includes(asset.id));
  const ids = choice === "pdd" ? [...STANDARD_MODEL_IDS, ...PDD_MODEL_IDS] : STANDARD_MODEL_IDS;
  return assets.filter((asset) => ids.includes(asset.id));
}

let configurationDownloading = false;
function downloadableAssets(assets) {
  return assets.filter((asset) => asset.downloadable && !["ready", "invalid", "downloading"].includes(modelAssetStatus(asset)));
}

async function downloadConfiguration() {
  if (configurationDownloading || !$("#modelLicenseConsent").checked || $("#modelConfiguration").value === "all") return;
  const assets = downloadableAssets(configurationAssets(state.modelSetup?.assets || []));
  if (!assets.length) return;
  configurationDownloading = true;
  renderModelSetup(state.modelSetup);
  try {
    await api("/api/models/setup/prepare", { method: "POST", body: {} });
    for (const asset of assets) {
      await api("/api/models/setup/download", {
        method: "POST",
        body: { asset_id: asset.id, license_accepted: true, source_urls: modelSourceUrls(asset.id) },
      });
    }
    toast(`${assets.length}件のモデル取得を受け付けました。各モデルの進捗を確認してください`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    configurationDownloading = false;
    await refreshModelSetup(true);
  }
}

function modelAssetCard(asset, consent) {
  const status = modelAssetStatus(asset);
  const size = asset.size_label || (asset.size_bytes ? bytes(asset.size_bytes) : "容量不明");
  const role = asset.role_label || MODEL_ROLE_LABELS[asset.role] || asset.role || "H3";
  const filename = asset.filename || asset.name || asset.label || asset.id;
  const path = asset.path || asset.relative_path || "標準フォルダ";
  const sourceUrl = typeof asset.source_url === "string" && asset.source_url.startsWith("https://") ? asset.source_url : "";
  const licenseUrl = typeof asset.license_url === "string" && asset.license_url.startsWith("https://") ? asset.license_url : "";
  const provenance = [
    sourceUrl ? `<a class="model-source" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noreferrer">公開元 ↗</a>` : "",
    licenseUrl ? `<a class="model-source" href="${escapeHTML(licenseUrl)}" target="_blank" rel="noreferrer">規約 ↗</a>` : "",
  ].filter(Boolean).join(" · ");
  const canDownload = Boolean(asset.downloadable) && !["ready", "invalid", "downloading"].includes(status) && consent;
  const actionLabel = status === "ready" ? "準備済み" : status === "present" ? "検証" : status === "downloading" ? "取得中…" : status === "invalid" ? "要確認" : status === "retry" ? "再試行" : asset.downloadable === false ? "手動配置" : "ダウンロード";
  const actionTitle = !consent && asset.downloadable && !["ready", "downloading"].includes(status) ? "ライセンス確認にチェックすると取得できます" : "";
  const rawProgress = Number(asset.progress ?? asset.download_progress);
  const progress = status === "downloading" && Number.isFinite(rawProgress)
    ? `<div class="model-progress"><progress max="100" value="${Math.max(0, Math.min(100, rawProgress))}"></progress></div>` : "";
  const elapsed = asset.setup_elapsed_seconds;
  const timing = elapsed != null ? `${asset.download_finished_at ? "取得・検証時間" : "経過"} ${clock(elapsed)}` : "";
  const transferred = status === "downloading" && asset.bytes_total ? ` · ${bytes(asset.bytes_downloaded)} / ${bytes(asset.bytes_total)}` : "";
  const speed = status === "downloading" && asset.download_speed_bytes_per_sec > 0 ? ` · ${bytes(asset.download_speed_bytes_per_sec)}/s` : "";
  const eta = status === "downloading" && asset.download_eta_seconds != null ? ` · 残り約 ${clock(asset.download_eta_seconds)}` : "";
  const detailText = `${timing}${transferred}${speed}${eta}`.replace(/^ · /, "");
  const downloadDetail = detailText ? `<span class="model-download-detail">${escapeHTML(detailText)}</span>` : "";
  const notes = Array.isArray(asset.notes) && asset.notes.length
    ? `<span class="model-asset-note">${escapeHTML(asset.notes.join(" / "))}</span>` : "";
  const experimental = asset.experimental ? '<span class="model-asset-experimental">EXPERIMENTAL</span>' : "";
  const sourceFields = (asset.source_files || []).map((file) => {
    const saved = state.modelSourceUrls[asset.id]?.[file.remote_path] || "";
    return `<label class="model-source-field"><span>${escapeHTML(file.remote_path)}</span>
      <input type="url" inputmode="url" spellcheck="false" autocomplete="off"
        data-model-source-asset="${escapeHTML(asset.id)}" data-model-source-file="${escapeHTML(file.remote_path)}"
        value="${escapeHTML(saved)}" placeholder="${escapeHTML(file.source_url)}"
        aria-label="${escapeHTML(file.remote_path)} の代替URL"></label>`;
  }).join("");
  const sourceEditor = status === "ready" || !sourceFields ? "" : `<details class="model-source-editor" data-model-source-editor="${escapeHTML(asset.id)}" ${state.openModelSources.has(asset.id) ? "open" : ""}>
    <summary>取得元URLを変更 <small>リンク切れの場合</small></summary>
    <p>Hugging FaceのファイルURLを入力。空欄なら標準の取得元を使います。サイズとSHA-256が一致したファイルだけ配置します。</p>
    ${sourceFields}</details>`;
  return `<article class="model-asset is-${status}${asset.experimental ? " experimental" : ""}" data-model-id="${escapeHTML(asset.id || "")}">
    <div class="model-asset-main">
      <div class="model-asset-title"><span class="model-asset-role">${escapeHTML(role)}</span>${experimental}<b title="${escapeHTML(filename)}">${escapeHTML(asset.label || filename)}</b></div>
      <span class="model-asset-meta" title="${escapeHTML(path)}">${escapeHTML(filename)} · ${escapeHTML(size)}${provenance ? ` · ${provenance}` : ""}</span>
      <span class="model-asset-status ${status}">${escapeHTML(modelAssetStatusLabel(asset, status))}</span>${downloadDetail}${notes}${progress}${sourceEditor}
    </div>
    <button type="button" class="secondary-button model-asset-action" data-model-download="${escapeHTML(asset.id || "")}" ${canDownload ? "" : "disabled"} ${actionTitle ? `title="${escapeHTML(actionTitle)}"` : ""}>${actionLabel}</button>
  </article>`;
}

function renderModelSetup(payload) {
  state.modelSetup = payload || {};
  updateModelAvailability();
  const allAssets = payload?.assets || payload?.models || [];
  const assets = configurationAssets(allAssets);
  const choice = $("#modelConfiguration").value;
  const total = assets.reduce((sum, asset) => sum + (Number(asset.size_bytes) || 0), 0);
  const readyBytes = assets.reduce((sum, asset) => {
    const status = modelAssetStatus(asset);
    if (status === "ready") return sum + (Number(asset.size_bytes) || 0);
    if (status === "downloading") return sum + (Number(asset.bytes_downloaded) || 0);
    return sum;
  }, 0);
  const remaining = assets.filter((asset) => modelAssetStatus(asset) !== "ready").reduce((sum, asset) => sum + (Number(asset.size_bytes) || 0), 0);
  const consent = $("#modelLicenseConsent").checked;
  const pending = downloadableAssets(assets);
  const configurationNotes = {
    comfy: `FastH3用${assets.length}ファイル · 合計 ${bytes(total)} · 未準備 ${bytes(remaining)}。TEXT・VAEは標準構成と共有します。取得後は生成画面で「FastH3 INT8」を選択してください。`,
    fl2va: `画像から生成するための${assets.length}ファイル · 合計 ${bytes(total)} · 未準備 ${bytes(remaining)}。取得後は生成画面で「画像優先 FL2VA」を選択してください。`,
    standard: `迷ったらこの構成。「高速・画質優先」で使う通常H3の必要セットです。必要ファイル ${assets.length}件・合計 ${bytes(total)}（未準備 ${bytes(remaining)}）。取得後は「標準パスを適用」を押してください。`,
    pdd: `標準セットに8-Step用の追加モデルを含みます。品質は標準構成と比較してください。必要ファイル ${assets.length}件・合計 ${bytes(total)}（未準備 ${bytes(remaining)}）。`,
    all: "実験用・別エンジン用も含みます。すべてのモデルを取得する必要はありません。",
  };
  $("#modelConfigurationNote").textContent = configurationNotes[choice] || "";
  const button = $("#downloadConfiguration");
  button.disabled = configurationDownloading || choice === "all" || !consent || !pending.length;
  button.textContent = configurationDownloading
    ? "取得を開始しています…"
    : pending.length ? `未取得の${pending.length}件をまとめて取得` : "この構成はすべて準備済み";
  button.title = !consent && pending.length ? "ライセンス確認にチェックすると取得できます" : "";

  const meter = $("#configMeter");
  meter.hidden = choice === "all" || !total;
  if (!meter.hidden) {
    const percent = Math.min(100, (100 * readyBytes) / total);
    $("#configProgress").value = percent;
    $("#configProgressText").textContent = `${Math.floor(percent)}% · ${bytes(readyBytes)} / ${bytes(total)}`;
  }

  const ready = assets.filter((asset) => modelAssetStatus(asset) === "ready").length;
  const activeCount = assets.filter((asset) => modelAssetStatus(asset) === "downloading").length;
  $("#modelSetupSummary").textContent = assets.length
    ? `${ready} / ${assets.length} 準備済み${activeCount ? ` · ${activeCount}件ダウンロード中` : ""}`
    : "モデル未確認";
  const list = $("#modelAssetList");
  // Keep an alternate-URL field usable while the 2 s progress poll runs.
  const editing = list.contains(document.activeElement) && document.activeElement?.matches?.("input");
  if (!editing) {
    list.innerHTML = assets.length
      ? assets.map((asset) => modelAssetCard(asset, consent)).join("")
      : '<div class="model-setup-empty">モデルカタログを読み込めませんでした。再スキャンしてください。</div>';
  }
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
  if (Array.isArray(payload.fast_profile_issues) && payload.fast_profile_issues.length) {
    lines.push(...payload.fast_profile_issues.map((item) => `・Blackwell最速条件: ${item}`));
  } else if (payload.fast_profile_ready === true) {
    lines.push("Blackwell最速プロファイル: 利用可能（sm100a VSA + FA4 + regional compile）");
  }
  element.className = `fasth3-status ${payload.ready ? "ready" : "blocked"}`;
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
    toast(payload.ready ? "FastH3の利用条件を満たしています" : "FastH3はまだ利用条件を満たしていません", payload.ready ? "success" : "info");
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
    state.modelSetupLoaded = true;
    renderModelSetup(payload);
  } catch (error) {
    state.modelSetupLoaded = true;
    $("#modelSetupSummary").textContent = "確認できません";
    state.modelSetup = null;
    updateModelAvailability();
    $("#modelAssetList").innerHTML = `<div class="model-setup-empty">${escapeHTML(error.message)}</div>`;
    if (!silent) toast(error.message, "error");
  }
}

async function withBusyButton(button, task) {
  button.disabled = true;
  try { return await task(); } finally { button.disabled = false; }
}

async function prepareModelDirs() {
  await withBusyButton($("#prepareModelDirs"), async () => {
    try {
      const result = await api("/api/models/setup/prepare", { method: "POST", body: {} });
      if (result.settings && state.bootstrap) {
        state.bootstrap.settings = result.settings;
        renderSettings(result.settings, result.readiness, result.openai);
      }
      await refreshModelSetup(true);
      toast("標準モデルフォルダを準備しました");
    } catch (error) { toast(error.message, "error"); }
  });
}

async function applyStandardPaths() {
  const ok = await confirmAction({
    eyebrow: "APPLY STANDARD PATHS",
    title: "HAYATE標準のモデルパスを設定に反映しますか？",
    body: "モデル定義・checkpoint・PDD・FastVideoのパスが標準フォルダに置き換わります。upstream・出力先・Pythonのパスは変更しません。",
    confirmLabel: "標準パスを適用",
  });
  if (!ok) return;
  await withBusyButton($("#applyStandardPaths"), async () => {
    try {
      const result = await api("/api/models/setup/apply-standard", { method: "POST", body: {} });
      if (state.bootstrap) state.bootstrap.settings = result.settings;
      renderSettings(result.settings, result.readiness, result.openai);
      toast("標準モデルパスを設定に反映しました");
      await refreshModelSetup(true);
    } catch (error) { toast(error.message, "error"); }
  });
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
      body: { asset_id: assetId, license_accepted: true, source_urls: modelSourceUrls(assetId) },
    });
    await refreshModelSetup(true);
    toast("モデルのダウンロードを開始しました");
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
  if (payload.openai_api_key && payload.clear_openai_api_key) {
    toast("APIキーの入力と消去は同時に指定できません", "error");
    return;
  }
  const button = $("#saveSettings");
  await withBusyButton(button, async () => {
    try {
      const result = await api("/api/settings", { method: "PUT", body: payload });
      if (state.bootstrap) {
        state.bootstrap.openai = result.openai;
        state.bootstrap.settings = result.settings;
      }
      renderSettings(result.settings, result.readiness, result.openai);
      toast("エンジン設定を保存しました");
      refreshJobs();
    } catch (error) { toast(error.message, "error"); }
  });
}

function renderHardware(hardware) {
  const gpu = hardware?.gpus?.find((item) => item.selected_for_inference) || hardware?.gpus?.[0];
  if (gpu) $("#gpuName").textContent = gpu.name;
  renderGpuSelectors(hardware, state.bootstrap?.settings?.gpu_default_selector || "auto");
  const gpuRows = (hardware?.gpus || []).map((item) => `<div class="hardware-item"><span>GPU ${item.index}${item.selected_for_inference ? " · primary" : ""}</span><b>${escapeHTML(item.name)} · ${bytes(item.vram_total_bytes)}${item.compute_capability ? ` · SM ${escapeHTML(item.compute_capability)}` : ""}${item.h3_eligible === false ? `<small>H3対象外: ${escapeHTML(item.eligibility_reason || "")}</small>` : ""}${item.uuid ? `<small>${escapeHTML(item.uuid)}</small>` : ""}</b></div>`).join("");
  $("#hardwarePanel").innerHTML = `<h3>Hardware</h3>
    <div class="hardware-item"><span>CPU</span><b>${escapeHTML(hardware?.cpu || "—")}</b></div>
    ${gpuRows || '<div class="hardware-item"><span>GPU</span><b>検出できません</b></div>'}
    <div class="hardware-item"><span>スケジューリング</span><b>${hardware?.auto_assignable_gpu_count ?? hardware?.eligible_gpu_count ?? 0} / ${hardware?.gpu_count || 0} 台を自動割当<small>${hardware?.multi_gpu_supported ? "multi-GPU scheduling available" : "single adapter"}</small></b></div>
    <div class="hardware-item"><span>PyTorch / CUDA</span><b>${escapeHTML(hardware?.pytorch_version || "—")}<small>CUDA ${escapeHTML(hardware?.pytorch_cuda_version || "—")}</small></b></div>`;
}

async function pollResources() {
  if (state.resourcePolling || document.hidden) return;
  state.resourcePolling = true;
  try {
    const resources = await api("/api/system");
    const activeJob = findJob(state.activeJobId);
    const gpu = resources.gpus?.find((item) => item.index === activeJob?.assigned_gpu_index)
      || resources.gpus?.[0];
    if (gpu) {
      const percent = gpu.total_bytes ? (100 * gpu.used_bytes) / gpu.total_bytes : 0;
      $("#vramText").textContent = `${bytes(gpu.used_bytes)} / ${bytes(gpu.total_bytes)}`;
      $("#vramMeter").value = percent;
      $("#vramMeter").classList.toggle("high", percent > 92);
      $("#gpuName").textContent = resources.gpus.length > 1 ? `${gpu.name}（GPU ${gpu.index}）` : gpu.name;
      const load = [
        Number.isFinite(gpu.utilization_percent) ? `${Math.round(gpu.utilization_percent)}%` : "",
        Number.isFinite(gpu.temperature_c) ? `${Math.round(gpu.temperature_c)}°C` : "",
      ].filter(Boolean).join(" · ");
      $("#gpuLoad").textContent = load || "—";
    }
    $("#ramText").textContent = `${bytes(resources.ram_used_bytes)} / ${bytes(resources.ram_total_bytes)}`;
    $("#ramMeter").value = resources.ram_percent;
    $("#ramMeter").classList.toggle("high", resources.ram_percent > 92);
    state.resourceFailures = 0;
    if (state.connectionLost) {
      state.connectionLost = false;
      $("#machineDot").classList.remove("offline");
      toast("サーバーに再接続しました", "info");
      refreshJobs();
      refreshModelSetup(true);
    }
  } catch {
    state.resourceFailures += 1;
    if (state.resourceFailures >= 2 && !state.connectionLost) {
      state.connectionLost = true;
      $("#machineDot").classList.add("offline");
      toast("HAYATEサーバーとの接続が切れました。再接続を試みています…", "error");
    }
  } finally {
    state.resourcePolling = false;
    updateEngineStatus();
  }
}

/* ------------------------------------------------------------------ */
/* Events                                                              */
/* ------------------------------------------------------------------ */

function switchToCustom() {
  const custom = profileRadio("custom");
  state.preferredProfile = "custom";
  checkProfile(custom);
  updateProfileSummary();
  updateComfyControls();
}

function isEditable(target) {
  return Boolean(target?.closest?.("input, textarea, select, [contenteditable='true']"));
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
  document.addEventListener("click", (event) => {
    const go = event.target.closest("[data-go]");
    if (go) navigate(go.dataset.go);
  });
  window.addEventListener("hashchange", () => showView(hashView()));
  $("#enginePill").addEventListener("click", () => navigate("settings"));

  // Composer
  const form = $("#generationForm");
  form.addEventListener("submit", submitGeneration);
  form.addEventListener("input", saveDraft);
  form.addEventListener("change", saveDraft);
  $("#prompt").addEventListener("input", updatePromptCount);
  $("#structurePrompt").addEventListener("click", openPromptAssist);
  ["#promptVisual", "#promptSound", "#promptMusic"].forEach((selector) => {
    $(selector).addEventListener("input", structuredPromptDraft);
  });
  $("#applyStructuredPrompt").addEventListener("click", applyStructuredPrompt);
  $("#generateAIPrompt").addEventListener("click", generateAIPrompt);
  $("#promptBrief").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); generateAIPrompt(); }
  });
  $("#promptDialog").addEventListener("close", () => {
    state.dialogTrigger?.focus?.();
    state.dialogTrigger = null;
  });
  $$("[data-prompt-chip]").forEach((button) => button.addEventListener("click", () => {
    const prompt = $("#prompt");
    const chip = button.dataset.promptChip;
    const start = prompt.selectionStart ?? prompt.value.length;
    const end = prompt.selectionEnd ?? prompt.value.length;
    const before = prompt.value.slice(0, start);
    const insert = `${before.trim() && !before.endsWith("\n\n") ? (before.endsWith("\n") ? "\n" : "\n\n") : ""}${chip}`;
    prompt.setRangeText(insert, start, end, "end");
    prompt.focus();
    prompt.dispatchEvent(new Event("input", { bubbles: true }));
  }));
  $("#samplePrompt").addEventListener("click", () => {
    const previous = $("#prompt").value;
    const sample = state.imageAsset ? I2V_SAMPLE_PROMPT : SAMPLE_PROMPT;
    state.promptTransform = null;
    setPrompt(sample);
    updatePromptTransformNote();
    saveDraft();
    $("#prompt").focus();
    if (previous.trim() && previous !== sample) {
      toast("サンプルを入力しました", "info", { action: { label: "元に戻す", run: () => { setPrompt(previous); saveDraft(); } } });
    }
  });
  $("#clearPrompt").addEventListener("click", () => {
    const previous = $("#prompt").value;
    const previousTransform = state.promptTransform;
    if (!previous) return;
    state.promptTransform = null;
    setPrompt("");
    updatePromptTransformNote();
    saveDraft();
    $("#prompt").focus();
    toast("プロンプトをクリアしました", "info", {
      action: { label: "元に戻す", run: () => { setPrompt(previous); state.promptTransform = previousTransform; updatePromptTransformNote(); saveDraft(); } },
    });
  });
  $("#revertPrompt").addEventListener("click", () => {
    if (!state.promptTransform) return;
    setPrompt(state.promptTransform.original);
    state.promptTransform = null;
    updatePromptTransformNote();
    saveDraft();
    toast("元の文章に戻しました", "info");
  });

  $$('input[name="profile"]').forEach((radio) => radio.addEventListener("change", () => {
    state.preferredProfile = radio.value;
    checkProfile(radio);
    applyProfile(radio.value);
    updateModelAvailability();
    if (radio.value === "custom") $("#advancedSettings").open = true;
  }));
  ADVANCED_CONTROL_IDS.forEach((id) => {
    $(`#${id}`).addEventListener("change", switchToCustom);
  });
  $("#easycache").addEventListener("change", () => {
    if ($("#easycache").checked) $("#pdd").checked = false;
    updateProfileSummary();
  });
  $("#pdd").addEventListener("change", () => {
    if ($("#pdd").checked) {
      $("#easycache").checked = false;
      $("#steps").value = 9;
    }
    updateProfileSummary();
  });
  ["vsaKeep", "fastVaeBatch"].forEach((id) => $(`#${id}`).addEventListener("change", updateComfyControls));
  ["task", "gpuDevice", "seed"].forEach((id) => $(`#${id}`).addEventListener("change", updateProfileSummary));
  $("#seedRandom").addEventListener("change", updateSeedMode);
  $("#randomSeed").addEventListener("click", () => {
    $("#seedRandom").checked = false;
    $("#seed").value = Math.floor(Math.random() * 2147483647);
    updateSeedMode();
    saveDraft();
  });

  $$("#durationControl button").forEach((button) => button.addEventListener("click", () => {
    setDuration(Number(button.dataset.duration));
    saveDraft();
  }));
  $("#customDuration").addEventListener("input", (event) => {
    const raw = event.target.value;
    if (raw === "") return;
    const value = Number(raw);
    if (value >= 1 && value <= 30) setDuration(value, { fromCustom: true });
    else event.target.setAttribute("aria-invalid", "true");
  });
  $("#customDuration").addEventListener("change", (event) => {
    if (event.target.value === "") { setDuration(state.duration); return; }
    setDuration(Number(event.target.value));
  });
  $("#resolutionPreset").addEventListener("change", updateResolution);
  ["#width", "#height"].forEach((selector) => {
    $(selector).addEventListener("input", updateResolution);
    $(selector).addEventListener("change", (event) => {
      event.target.value = snapDimension(event.target.value);
      updateResolution();
    });
  });

  [false, true].forEach((last) => {
    const prefix = last ? "lastImage" : "image";
    $(`#${prefix}File`).addEventListener("change", (event) => uploadImage(event.target.files?.[0], last));
    $(last ? "#removeLastImage" : "#removeImage").addEventListener("click", () => removeImage(last));
    $(`#${prefix}Preview`).addEventListener("load", (event) => {
      state.imageSizes[last ? "last" : "first"] = { width: event.target.naturalWidth, height: event.target.naturalHeight };
      updateImageTools();
    });
    const zone = $(last ? "#lastDropzone" : "#dropzone");
    ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => {
      event.preventDefault();
      if (!$(`#${prefix}File`).disabled) zone.classList.add("dragging");
    }));
    ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("dragging"); }));
    zone.addEventListener("drop", (event) => {
      if ($(`#${prefix}File`).disabled) return;
      uploadImage(event.dataTransfer.files?.[0], last);
    });
  });
  $("#swapImages").addEventListener("click", swapImages);
  $("#matchImageResolution").addEventListener("click", matchImageResolution);
  document.addEventListener("paste", (event) => {
    if (state.view !== "generate" || document.querySelector("dialog[open]")) return;
    const file = [...(event.clipboardData?.files || [])].find((item) => IMAGE_TYPES.includes(item.type));
    if (!file) return;
    if ($("#imageFile").disabled) {
      toast("このプロファイルは画像入力に対応していません", "error");
      return;
    }
    event.preventDefault();
    uploadImage(file, Boolean(state.imageAsset && !state.lastImageAsset));
  });

  // Live panel
  $("#stopSaveButton").addEventListener("click", () => state.activeJobId && stopJob(state.activeJobId, true));
  $("#cancelButton").addEventListener("click", () => state.activeJobId && stopJob(state.activeJobId, false));
  $("#liveOpenButton").addEventListener("click", () => state.activeJobId && openJob(state.activeJobId));
  $("#liveReuseButton").addEventListener("click", () => state.activeJobId && reuseJob(state.activeJobId));
  $("#liveDismissButton").addEventListener("click", () => { state.activeJobId = null; renderLive(null); });

  // Global toolbar
  $("#refreshButton").addEventListener("click", async () => {
    const button = $("#refreshButton");
    button.classList.add("spinning");
    await Promise.all([refreshJobs(), pollResources(), refreshModelSetup(true)]);
    button.classList.remove("spinning");
    toast("最新情報に更新しました", "info");
  });
  $("#themeButton").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "clear" : "dark");
  });

  // Settings
  $("#saveSettings").addEventListener("click", saveSettings);
  $("#view-settings").addEventListener("input", (event) => { if (event.target.matches("[data-setting]")) markSettingsDirty(true); });
  $("#view-settings").addEventListener("change", (event) => { if (event.target.matches("[data-setting]")) markSettingsDirty(true); });
  $("#prepareModelDirs").addEventListener("click", prepareModelDirs);
  $("#applyStandardPaths").addEventListener("click", applyStandardPaths);
  $("#refreshModelSetup").addEventListener("click", async () => {
    await withBusyButton($("#refreshModelSetup"), () => refreshModelSetup());
    toast("モデル状態を再スキャンしました", "info");
  });
  $("#checkFastH3").addEventListener("click", checkFastH3);
  $("#modelConfiguration").addEventListener("change", () => renderModelSetup(state.modelSetup));
  $("#downloadConfiguration").addEventListener("click", downloadConfiguration);
  $("#modelLicenseConsent").addEventListener("change", () => renderModelSetup(state.modelSetup));
  $("#modelAssetList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-model-download]");
    if (button) downloadModel(button.dataset.modelDownload);
  });
  $("#modelAssetList").addEventListener("input", (event) => {
    const field = event.target.closest("[data-model-source-asset]");
    if (!field) return;
    const assetId = field.dataset.modelSourceAsset;
    (state.modelSourceUrls[assetId] ||= {})[field.dataset.modelSourceFile] = field.value;
  });
  $("#modelAssetList").addEventListener("toggle", (event) => {
    const editor = event.target.closest("[data-model-source-editor]");
    if (!editor) return;
    if (editor.open) state.openModelSources.add(editor.dataset.modelSourceEditor);
    else state.openModelSources.delete(editor.dataset.modelSourceEditor);
  }, true);
  window.addEventListener("beforeunload", (event) => {
    if (!state.settingsDirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  // Library & queue
  $("#librarySearch").addEventListener("input", renderLibrary);
  $("#libraryFilter").addEventListener("change", renderLibrary);
  $("#librarySort").addEventListener("change", renderLibrary);
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-delete-job], [data-reuse-job], [data-open-job], [data-live-job], [data-stop-job], [data-cancel-job]");
    if (!target || target.disabled) return;
    const { deleteJob: del, reuseJob: reuse, openJob: open, liveJob, stopJob: stop, cancelJob } = target.dataset;
    if (del) deleteJob(del);
    else if (reuse) reuseJob(reuse);
    else if (open) openJob(open);
    else if (liveJob) { state.activeJobId = liveJob; renderLive(findJob(liveJob)); navigate("generate"); }
    else if (stop) stopJob(stop, true);
    else if (cancelJob) stopJob(cancelJob, false);
  });
  const grid = $("#libraryGrid");
  grid.addEventListener("mouseover", (event) => {
    const thumb = event.target.closest(".thumb");
    const video = thumb && $("video", thumb);
    if (!video || thumb.contains(event.relatedTarget)) return;
    loadVideo(video);
    video.play().catch(() => {});
  });
  grid.addEventListener("mouseout", (event) => {
    const thumb = event.target.closest(".thumb");
    const video = thumb && $("video", thumb);
    if (!video || thumb.contains(event.relatedTarget)) return;
    video.pause();
    try { video.currentTime = 0.1; } catch { /* metadata not ready */ }
  });

  // Job dialog
  $("#closeDialog").addEventListener("click", () => $("#videoDialog").close());
  $("#videoDialog").addEventListener("click", (event) => { if (event.target === event.currentTarget) event.currentTarget.close(); });
  $("#videoDialog").addEventListener("close", () => {
    const video = $("#dialogVideo");
    video.pause();
    video.removeAttribute("src");
    video.load();
    state.dialogJobId = null;
    if (state.dialogTrigger?.isConnected) state.dialogTrigger.focus?.();
    state.dialogTrigger = null;
  });
  $("#dialogLogBox").addEventListener("toggle", (event) => { if (event.target.open) loadDialogLog(); });
  $("#deleteDialogJob").addEventListener("click", () => state.dialogJobId && deleteJob(state.dialogJobId));
  $("#reuseDialogJob").addEventListener("click", () => state.dialogJobId && reuseJob(state.dialogJobId));
  $("#copyDialogPrompt").addEventListener("click", async () => {
    const text = $("#dialogPrompt").textContent;
    try {
      await navigator.clipboard.writeText(text);
      toast("プロンプトをコピーしました", "info");
    } catch {
      const range = document.createRange();
      range.selectNodeContents($("#dialogPrompt"));
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      toast("コピーできませんでした。選択したテキストを Ctrl+C でコピーしてください", "error");
    }
  });

  // Keyboard
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && state.view === "generate" && !document.querySelector("dialog[open]")) {
      submitGeneration(event);
      return;
    }
    if (event.altKey && !event.ctrlKey && !event.metaKey && VIEW_KEYS[event.key]) {
      event.preventDefault();
      navigate(VIEW_KEYS[event.key]);
    }
    if (event.key === "/" && !isEditable(event.target) && state.view === "library") {
      event.preventDefault();
      $("#librarySearch").focus();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    updateDocumentTitle(findJob(state.activeJobId));
    pollResources();
  });
}

async function initialize() {
  applyTheme(savedTheme(), false);
  bindEvents();
  restoreDraft();
  updatePromptCount();
  updateDuration();
  updateResolution();
  updateSeedMode();
  updatePromptTransformNote();
  applyProfile(selectedProfile());
  showView(hashView());
  renderLive(null);
  updateSubmitState();
  try {
    const bootstrap = await api("/api/bootstrap");
    state.bootstrap = bootstrap;
    state.jobs = bootstrap.jobs || [];
    $("#version").textContent = bootstrap.version;
    renderNetwork(bootstrap.network);
    renderSettings(bootstrap.settings, bootstrap.readiness, bootstrap.openai);
    renderHardware(bootstrap.hardware);
    applyProfile(selectedProfile());
    renderJobs();
    const activeJobs = state.jobs.filter(isActive);
    activeJobs.forEach((job) => connectEvents(job.id));
    const live = activeJobs.find((job) => job.status !== "queued") || activeJobs[activeJobs.length - 1];
    if (live) { state.activeJobId = live.id; renderLive(live); }
  } catch (error) {
    state.bootstrapError = true;
    toast(error.message, "error");
  }
  await refreshModelSetup(true);
  await pollResources();
  state.resourceTimer = setInterval(pollResources, 3000);
  state.elapsedTimer = setInterval(() => {
    const job = findJob(state.activeJobId);
    if (isActive(job) && job.started_at) $("#elapsedTime").textContent = clock(activeElapsed(job));
  }, 1000);
}

initialize();
