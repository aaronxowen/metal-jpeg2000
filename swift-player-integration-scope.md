# Scope: hybrid GPU decode integration into swift-dcp-player

**Goal:** Realtime 2K DCP playback on the **M1 Mac mini** by splitting JPEG 2000 decode across CPU and
GPU. Per `FEASIBILITY.md` §16, the decided architecture is a **hybrid**: the CPU runs Tier-2 + Tier-1
(the serial MQ coder, faster on CPU than GPU), the GPU runs the back-end (inverse 9/7 DWT + inverse ICT
+ DC level-shift + xyz_to_rgb), overlapped across frames via unified-memory zero-copy.

This is **integration engineering**, not feasibility — correctness and per-stage performance are
already proven (T1 bit-exact, back-end integer-exact, validated on M5 Pro + M1).

## Target architecture

```
CPU (decode thread)                       GPU (Metal)                     Display
──────────────────────                    ──────────────────────────     ─────────────
MXF read → AES decrypt → T2 → T1   ───►    iDWT → ICT → level-shift        MetalVideoView
(per frame, buffers REUSED)                → xyz_to_rgb → RGBA texture ───► samples texture
       frame N+1                    ‖         frame N (overlapped)          (no CPU readback)
```

Steady-state frame time ≈ `max(CPU: T2+T1, GPU: back-end + color)`. Both sides measured at ~37 ms on
the M1, balanced right at the 41.6 ms / 24fps budget — hence "tight but plausible" (§16).

## What we reuse
- **Validated Metal kernels** (`proto/`): `dwt_kernel.metal` (H/V), `backend_kernel.metal`
  (`backend_finalize` = ICT + level-shift) — integer-exact vs OpenJPEG.
- **Known split point:** the `OPJ_BACKEND_DUMP` hook location in `tcd.c` `opj_tcd_decode_tile`
  (between `opj_tcd_t1_decode` and `opj_tcd_dwt_decode`) is exactly the CPU/GPU boundary.
- **Player scaffolding** (swift-dcp-player): `DecodeWorker` (decode thread), `FrameQueue` (pipeline
  buffering), `MetalVideoView` (Metal renderer), `setDecodeReduction()` (the `reduce` fallback).

## The real challenge
The player decodes through dcpomatic's **Butler** facade
(`butler->get_video(BLOCKING)` → `PlayerVideo.image(RGBA)`, `PlayerEngine.cpp`), an all-in-one
"give me RGBA" path that does the full decode + XYZ→RGB internally. The hard part is **splitting that
facade** so decode stops after T1 and exposes the coefficient buffers — not the Metal (kernels done).

## Phased plan

### Phase A — Standalone hybrid harness (this branch; no libdcp/player)
The make-or-break test, isolated from Butler. Answers: *does buffer-reused CPU-T1 ‖ GPU-back-end
sustain 24fps on the M1?* — for ~10% of the effort of full player integration.
1. **`opj` decode-to-T1 entry point** in the fork: decode a J2K codestream through Tier-1 only and
   expose post-T1 per-component buffers + geometry (`boxes`, `dc_shift`, `mct`, `prec`). Productionizes
   the `tcd.c` split. Reuse-friendly (keep codec/tile buffers alive across same-size frames).
   *Validation:* its output must equal the input section of `backend_corpus.bin`.
2. **Swift orchestrator:** 2-thread pipeline, double-buffered, CPU T1 (buffers reused) ‖ GPU back-end
   (existing kernels), over a sequence of `.j2c` frames.
3. **Measure sustained fps on the M1**; validate integer-exact; confirm buffer reuse pulls the CPU
   side under ~41.6 ms.

> **Decision gate:** if Phase A does not sustain 24fps on the M1, stop and reconsider (optimize the
> T1/back-end kernels, use E-cores, `reduce` fallback, or rescope) **before** any player surgery.

### Phase B — GPU xyz_to_rgb
Fold XYZ→RGB (DCI 2.6 gamma → display colorspace) into the GPU back-end → RGBA texture. Validate vs
libdcp `xyz_to_rgb`. Removes another CPU stage and the final upload.

### Phase C — Player integration
Bypass the Butler for *picture* (keep it for audio + timing): drive the hybrid decoder from libdcp's
J2K frame readers + KDM decrypt, output a Metal texture straight to `MetalVideoView`. Honor the
seek/stop single-decode-thread contract, stereo, and `reduce`.

## Open decisions / risks
- **Which OpenJPEG the player links** — must become this fork (with the decode-to-T1 entry); the
  Engine currently ships prebuilt libs.
- **Zero-copy vs cheap copy** of post-T1 buffers — a ~26 MB/frame memcpy is ~0.5 ms on unified memory;
  acceptable if having opj write into shared `MTLBuffer`s is awkward.
- **Buffer reuse is load-bearing** for the budget — keep the opj codec + buffers alive across frames
  (libdcp recreates per frame today).
- **Confirm mono (2D) playback** — stereo halves the budget.
- 4K / HFR are out of reach with this architecture (CPU MQ-coder wall on 4 P-cores).
