param(
    [string]$ComfyRoot = "M:\Project\HAYATE-ComfyUI",
    [string]$Listen = "0.0.0.0",
    [int]$Port = 8189,
    [switch]$EnableTurbo,
    [switch]$EnableSageAttention,
    [switch]$EnableComfyKitchenAttention,
    [switch]$EnableTritonBackend,
    [switch]$EnableFastOptimizations,
    [switch]$EnableAllFastOptimizations,
    [switch]$EnablePinnedMemory,
    [switch]$EnableFastDisk,
    [int]$AsyncOffloadStreams = 2
)

$ErrorActionPreference = "Stop"

if ($EnableTurbo) {
    if ($EnableComfyKitchenAttention -or $EnableFastOptimizations -or $EnableFastDisk -or $EnableSageAttention -or $EnableAllFastOptimizations -or $EnablePinnedMemory -or $EnableTritonBackend -or $AsyncOffloadStreams -ne 2) {
        throw "-EnableTurbo sets SageAttention, Triton, all --fast, pinned memory, and 4 offload streams; do not combine it with other performance switches."
    }
    $EnableSageAttention = $true
    $EnableTritonBackend = $true
    $EnableAllFastOptimizations = $true
    $EnablePinnedMemory = $true
    $AsyncOffloadStreams = 4
}

if ($EnableSageAttention -and $EnableComfyKitchenAttention) {
    throw "Choose one attention backend: -EnableSageAttention or -EnableComfyKitchenAttention."
}
if ($EnableFastOptimizations -and $EnableAllFastOptimizations) {
    throw "Choose one --fast mode: -EnableFastOptimizations or -EnableAllFastOptimizations."
}
if ($EnableFastDisk -and $EnablePinnedMemory) {
    throw "-EnableFastDisk and -EnablePinnedMemory are mutually exclusive."
}

$python = Join-Path $ComfyRoot ".venv\Scripts\python.exe"
$main = Join-Path $ComfyRoot "main.py"
$sharedRoot = "M:\Project\HAYATE"
$outputRoot = Join-Path $sharedRoot "outputs"
$inputRoot = Join-Path $sharedRoot "inputs"
$userRoot = Join-Path $sharedRoot "data\comfyui-user"
$cacheRoot = Join-Path $ComfyRoot ".cache"

foreach ($path in @($python, $main)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "ComfyUI installation is incomplete: $path"
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot, $inputRoot, $userRoot, $cacheRoot | Out-Null

# Keep dependency/model caches and generated files off the system drive.  The
# model registry itself is loaded from ComfyUI's extra_model_paths.yaml and
# points back to M:\Project\HAYATE\models without copying any weight files.
$env:TEMP = Join-Path $cacheRoot "tmp"
$env:TMP = $env:TEMP
$env:HF_HOME = Join-Path $sharedRoot "models\.cache\huggingface"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:TRITON_CACHE_DIR = Join-Path $cacheRoot "triton"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $cacheRoot "torchinductor"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $env:TEMP, $env:HF_HOME, $env:TRANSFORMERS_CACHE, $env:UV_CACHE_DIR, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR | Out-Null

$arguments = @(
    $main,
    "--listen", $Listen,
    "--port", $Port,
    "--disable-auto-launch",
    "--enable-dynamic-vram",
    "--output-directory", $outputRoot,
    "--input-directory", $inputRoot,
    "--user-directory", $userRoot
)
if (-not $EnablePinnedMemory) {
    $arguments += "--disable-pinned-memory"
}
if ($AsyncOffloadStreams -gt 0) {
    $arguments += @("--async-offload", $AsyncOffloadStreams)
}
if ($EnableFastDisk) {
    # Prefer the M: SSD as the dynamic-VRAM backing store.  This is useful
    # when pinned host memory would otherwise push a 32 GB machine into paging.
    $arguments += "--fast-disk"
}
if ($EnableSageAttention) {
    $arguments += "--use-sage-attention"
}
if ($EnableComfyKitchenAttention) {
    $arguments += "--use-ck-attention"
}
if ($EnableTritonBackend) {
    $arguments += "--enable-triton-backend"
}
if ($EnableFastOptimizations) {
    $arguments += @("--fast", "fp16_accumulation", "autotune")
}
if ($EnableAllFastOptimizations) {
    # ComfyUI's bare --fast enables the full set supported by this build.
    # Keep this opt-in because fp8/cublas features are experimental.
    $arguments += "--fast"
}

Write-Host "Starting HAYATE ComfyUI from $ComfyRoot" -ForegroundColor Cyan
Write-Host "Shared models: $sharedRoot\models" -ForegroundColor Green
Write-Host "Output: $outputRoot | Listen: $Listen`:$Port" -ForegroundColor Green
Write-Host ("Acceleration: attention={0}, triton={1}, pinned={2}, offload_streams={3}, fast_disk={4}" -f `
    ($(if ($EnableSageAttention) { "sage" } elseif ($EnableComfyKitchenAttention) { "comfy-kitchen" } else { "pytorch" }),
     [bool]$EnableTritonBackend, [bool]$EnablePinnedMemory, $AsyncOffloadStreams, [bool]$EnableFastDisk)) -ForegroundColor Yellow
& $python @arguments
exit $LASTEXITCODE
