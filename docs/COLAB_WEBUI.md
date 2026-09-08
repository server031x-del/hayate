# HAYATE Studio on Google Colab

The HAYATE WebUI can run inside a hosted Colab VM for UI and integration tests.
The local Windows machine is not used as a server.  A Colab VM is ephemeral, so
the source bundle, dependencies, processes, and downloaded weights must be
restored after a runtime disconnect.

## Source transfer

The repository is private.  The notebook therefore uses an explicit upload
instead of an unauthenticated Git clone:

```powershell
python scripts/create_colab_webui_bundle.py
```

Upload the resulting `colab-webui-bundle.zip` in the notebook's WebUI source
cell.  The archive contains the HAYATE package, WebUI static assets, settings,
and model catalog.  It does not contain model weights or secrets.

The next WebUI cell contains an opt-in model downloader.  Set
`DOWNLOAD_HAYATE_MODELS = True` and, after reviewing the model repository terms,
`ACCEPT_HAYATE_MODEL_LICENSE = True` to fetch the four direct HAYATE files.
They total about 29.8 GiB and are downloaded into the VM's
`/content/hayate-download-test` directory.  The cell checks the pinned revision,
exact byte size, and SHA-256 before the WebUI setup cell starts.  Leaving the
flag `False` still starts the UI and lets you use its model setup panel.

The following **FastH3 v1** cell is independent and also opt-in. Set
`INSTALL_FASTH3_RUNTIME = True` to add the pinned FastVideo 0.2.1 runtime to the
Colab environment, and set `DOWNLOAD_FASTH3_MODEL = True` plus
`ACCEPT_FASTH3_MODEL_LICENSE = True` to download the pinned
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` directory. The
snapshot is about 148 GB (about 138 GiB) and T2VA-only, so it is normally unsuitable for the
free T4 disk/runtime. The WebUI's strict FastH3 profile works on supported
SM80+ GPUs with Triton; the Blackwell最速 profile additionally requires the
sm100a kernel and `flash_attn.cute` (FA4).

## Colab layout

The setup cell extracts the archive into `/content/HAYATE-webui`, installs only
the `[webui]` extra into the current Colab Python environment, pins the audited
`maybleMyers/h3` checkout, and creates:

```text
/content/HAYATE-webui/
  configs/models.yaml
  hayate/webui/
  models/minimax-h3-snapshot/
  models/text_encoders/
  models/vae/
  models/lora/
  models/fastvideo/
  outputs/
  data/webui/
```

When the VM model set already exists under `/content/hayate-download-test`, the
notebook moves the HAYATE W4A8 transformer, text encoder, video VAE, and audio
VAE into the WebUI's own `models/` tree and leaves compatibility links at the
old VM paths.  This avoids a second copy while allowing HAYATE's model catalog
to verify the files safely.  Checkpoint support files are then obtained through
the WebUI catalog with the fixed revision and SHA-256 manifest.  The optional
FastVideo single-file artifact belongs to the separate ComfyUI/VSA route and is
not a HAYATE direct-generation input.

The WebUI can start without the optional PDD LoRA pair and reports that profile
as unavailable.  A free T4 runtime is for UI and preflight checks; HAYATE's
W4A8 generation requires a supported SM80+ GPU and sufficient host RAM/VRAM.

## Access

The server binds to `0.0.0.0:7860` and is never treated as reachable through
the browser's `localhost`.  The tunnel cell starts a temporary Cloudflare Quick
Tunnel and prints an HTTPS URL.  The URL is intended for a short test session;
the VM and URL disappear when the runtime is disconnected.  Use a named tunnel
or an authenticated reverse proxy before exposing a paid deployment.

## Model and billing boundary

The WebUI source/configuration can be validated on the free T4 runtime.  The
FastH3 VSA route remains disabled there because it requires a supported SM80+
GPU and more host RAM/VRAM than the free profile provides.  The optional
Blackwell最速 profile is stricter: it also requires sm100a VSA and FA4. Enable
the paid generation cells only after selecting a compatible runtime and
entering actual Colab billing values for the per-video cost report.

`colab/HAYATE.ipynb` keeps the HAYATE Studio WebUI path as its runnable flow;
the older FastH3/ComfyUI experiment remains in `colab/runtime.py` and is not
started by the WebUI cells.
