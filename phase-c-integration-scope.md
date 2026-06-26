# Phase C — detailed scope: hybrid decoder in swift-dcp-player

Companion to `swift-player-integration-scope.md`, grounded in the actual code. Goal: run the validated
hybrid decoder (CPU Tier-1 ‖ GPU back-end) as the picture path in the player and measure real sustained
playback fps on the M1. Phase A proved the architecture (28.9 fps, integer-exact, 7 ms margin); Phase C
makes it the player's decode path.

## Key finding — we intercept, not bypass (much cleaner than first assumed)

The player decodes via dcpomatic's **Butler**: `butler->get_video(BLOCKING)` → `PlayerVideo` →
`.image(RGBA)` does the full J2K→XYZ→RGB decode (`PlayerEngine.cpp:1082`). But:

- `PlayerVideo` holds a `shared_ptr<const ImageProxy> _in`, and for a DCP that proxy is a
  **`J2KImageProxy`**, which exposes the **decrypted compressed codestream** via a public
  `j2k() -> shared_ptr<const dcp::Data>` (`dcpomatic/src/lib/j2k_image_proxy.h:70`).
- So we **keep the Butler** (it still drives timing, audio, seek, 3D/eyes, frame sequencing, KDM key
  management). We only swap the *decode of the J2K bytes*. No reimplementing the timeline. The original
  "split the Butler" worry was pessimistic.

Decryption stays in libdcp/asdcp (hardware AES) — `j2k()` is already plaintext.

## Target data flow

```
Butler (timing/audio/seek/3D)                 Hybrid decoder (decode thread)        Display
─────────────────────────────                 ──────────────────────────────       ──────────────
get_video() -> PlayerVideo                     our-opj decode-to-T1 (CPU)           FrameQueue<MTLTexture>
  -> proxy() -> J2KImageProxy::j2k()  ──bytes─► GPU back-end (iDWT+ICT+shift)   ──► MetalVideoView
  (+ pts, eyes, reduce, size)                   GPU xyz_to_rgb -> RGBA texture        draws texture
```

Per-frame time ≈ `max(CPU T1, GPU back-end+color)`, overlapped frame N+1 ‖ frame N (the Phase A model).

## Component placement (data-driven)

| Stage | Where | Notes |
|---|---|---|
| MXF read + AES decrypt | libdcp (CPU) | unchanged; hardware AES, ~negligible |
| Tier-2 + **Tier-1** | CPU (decode thread) | our opj fork, `opj_set_t1_output_callback`, buffers reused |
| iDWT + ICT + level-shift | **GPU** | validated batched kernels (integer-exact) |
| **xyz_to_rgb** | **CPU — reuse dcpomatic's Image swscale** (revised again, see note) | the player's displayed color is FFmpeg `sws_scale` XYZ12LE→RGBA inside dcpomatic's `Image` (SWS_BICUBIC|SWS_ACCURATE_RND), NOT libdcp `xyz_to_rgba` and NOT a GPU kernel. DCP content *unsets* the colour conversion; `J2KImageProxy::prepare` only copies XYZ12. So feed the GPU's (integer-exact) XYZ into the same `Image` conversion → RGBA identical to the player, zero color-matching risk, CPU (M1 is GPU-bound). |
| timing / audio / seek / 3D | Butler | unchanged |

> Revision of §18: with the back-end now ~32 ms on M1 and a 7 ms margin, putting color on the GPU
> (which keeps a clean single GPU pass → display texture, no read-back/re-upload) is preferable to
> CPU color despite the GPU-bound profile. Re-evaluate against measured margin in C3.

## Build / link integration — the principal risk

The app **bundles opj inside libdcp** and does not link `openjp2` separately. To use our fork's
decode-to-T1 API in the Swift decoder, the options were:

- (a) Link our fork statically into the Swift target, symbol-isolated.
- (b) Rebuild the Engine stack (libdcp + dcpomatic) against our opj fork.

**DECISION (chosen): option (b)** — rebuild libdcp + dcpomatic against the `metal-jpeg2000` opj fork so
there is **one opj everywhere**, carrying the decode-to-T1 API. No symbol clash (single copy); libdcp's
own `decompress_j2k` and our Swift decoder share the same opj. Cleanest semantically.

Implications to plan for (C0):
- Build the fork's `libopenjp2` (+ headers), point **libdcp**'s build at it instead of the
  system/Homebrew openjpeg, rebuild libdcp, then rebuild **dcpomatic** (links libdcp). Replace the
  prebuilt `Engine/lib/libdcp-1.0` + `libdcpomatic2` with the rebuilt libs.
- Done on the swift-dcp-player **build host** (macOS 14 VM, `admin@192.168.64.2`) where the Engine libs
  are produced.
- Verify `nm` shows `opj_set_t1_output_callback` in the rebuilt libdcp's opj, and that the app still
  decodes a DCP normally before adding the hybrid path.

**Resolve this first (C0)** — it gates everything.

## Phased sub-steps

Built + validated in `swift-dcp-player` (branch `feature/metal-hybrid-decode`); full as-built record +
build/run commands are in that repo's `CLAUDE.md` ("GPU hybrid decode"). Done unless noted.

- **C0 — opj linking: option (b) — DONE.** Engine rebuilt against the fork into a separate
  `Engine-hybrid/` prefix (shared `Engine/` untouched); `build-engine.sh` LDFLAGS prefix-first +
  `-lopenjp2` + install_name fixup (fork-guarded); `Package.swift` auto-selects the prefix. DCP still
  decodes; the decode-to-T1 symbol is reachable in-app (opj 2.5.4). NOTE: built on the macOS-26 host
  for dev; a macOS-14-floor build must rebuild the fork opj on the VM too (it bakes the host OS).
- **C1 — expose the codestream — DONE.** `PlayerVideo::proxy()` getter (dcpomatic) + a probe pulling
  the decrypted `J2KImageProxy::j2k()` in `nextVideoFrame` (`KMQ_HYBRID_PROBE`). Verified live.
- **C2 — hybrid decoder — DONE (pixel-exact, in-player).** Split CPU/GPU rather than one Swift module:
  facade `decodeJ2KToT1` (opj memory-stream, **thread-local** decode-to-T1 hook) → `HybridJ2KBackend`
  (GPU iDWT+MCT+level-shift, **integer-exact** vs opj) → facade `xyzToRGBA` (reuse dcpomatic `Image`
  swscale, **pixel-exact** vs `nextVideoFrame`). Colour ended up CPU/swscale, not a GPU kernel (see
  the §xyz_to_rgb note).
- **C3 — wire into the pipeline — NEXT.** `FrameQueue` carries the hybrid frame (RGBA/texture);
  `MetalVideoView` draws it; decode thread overlaps CPU T1 ‖ GPU; honor the parked-on-seek contract.
  Play a real DCP; **measure sustained fps on the M1**. KEY PROBLEM: the hybrid path currently goes
  through `butler->get_video`, which makes the Butler full-decode every frame on its prefetch thread
  (double decode) — must avoid that to get the speedup (prepare/prefetch bypass; may touch Layer A).
- **C4 — finish.** `reduce` (proxy resolution factor → `opj_set_decoded_resolution_factor`; the engine
  already has `setDecodeReduction`), 3D/eyes (content is StereoVF; 2D playback picks one eye).

## Open questions / risks

- **opj symbol clash** (C0) — must resolve before anything else.
- **Color accuracy** — GPU xyz_to_rgb must match libdcp's pipeline (gamma 2.6 decode, XYZ→target
  matrix, display encode) within tolerance; validate in C2.
- **Margin under real load** — the 7 ms must absorb distinct-frame decode (vs the harness's warm
  same-frame loop), GPU color, texture/display, and A/V sync. First true reading comes in C3.
- **Non-DCP content** — the player also handles MPEG2/FFmpeg sources; the hybrid path is DCP-J2K only,
  everything else falls back to the existing `.image()` path.
- **4K / HFR** remain out of reach (≈4× both sides) — 2K@24 is the committed target.
