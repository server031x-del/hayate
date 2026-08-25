# Upstream license audit

Audit target: `https://github.com/maybleMyers/h3`, commit
`94220c1fdf14d6d9d40be06fb99f55b27c0d9024`.

## Findings

1. The repository has no top-level `LICENSE`, `LICENSE.md`, `COPYING`, or
   `NOTICE` file at the audited commit.
2. These MiniMax engine files carry explicit Apache-2.0 notices and can be
   considered for direct reuse with their notices retained:
   - `minimax_video/transformer.py`
   - `minimax_video/scheduler.py`
   - `minimax_video/vae_video.py`
   - `minimax_video/vae_audio.py`
   - `minimax_video/packing.py`
   - `minimax_video/packing_ref2va.py`
   - `minimax_video/qwen3vl_vision.py`
3. `minimax_video/sol_attn/` identifies its NVIDIA Sana source and Apache-2.0
   commit in `VENDORED.md`; its third-party notices must travel with any copy.
4. `GIMM-VFI/LICENSE` allows redistribution/use only for non-commercial
   purposes unless permission is obtained. It is excluded from HAYATE.
5. `modules/SeedVR/LICENSE` is Apache-2.0, but SeedVR is outside v0.1.
6. Important files including `pipeline.py`, `model_loader.py`,
   `conditioner.py`, `int8_quant.py`, `progressive_load.py`,
   `minimax_generate_video.py`, `wan_job_queue.py`, and `wan_worker.py` do not
   carry a clear per-file license in this checkout. Source comments describe
   upstream provenance for some code, but provenance is not itself a license.

## v0.1 policy

- HAYATE does not copy license-unclear upstream files.
- HAYATE may inspect them to define compatible extension interfaces and may
  execute a separately supplied local checkout through an adapter in a later
  milestone.
- Apache-2.0 files will only be vendored when they are actually needed; original
  notices, modifications, and attribution will be retained.
- `THIRD_PARTY_NOTICES.md` records the technical base even though v0.1 does not
  redistribute its code.
- Model weights have their own MiniMax license and are not covered by the
  HAYATE source-code license.

This is an engineering audit, not legal advice. Before distributing a build
that embeds the license-unclear files, obtain a repository-level license or
written permission from the upstream author.

