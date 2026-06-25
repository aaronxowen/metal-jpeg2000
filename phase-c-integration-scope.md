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
| **xyz_to_rgb** | **GPU** (revised from §18) | cheap LUT+matrix; outputs the display RGBA texture directly — no readback. ~few ms, fits the 7 ms margin. Fall back to CPU color only if margin tightens. |
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

- **C0 — opj linking: option (b)** (decided above). Rebuild libdcp + dcpomatic against the
  `metal-jpeg2000` opj fork on the build host; swap the prebuilt Engine libs; confirm a DCP still
  decodes normally and the decode-to-T1 symbol is present.
- **C1 — expose the codestream.** Add `PlayerVideo::proxy()` getter (1 line) in dcpomatic; add a
  `PlayerEngine` method returning the decrypted J2K bytes + pts + eyes + reduce + size per frame
  (dynamic_cast to `J2KImageProxy`; fall back to `.image()` for non-DCP content). Verify by logging
  codestream sizes and decoding one with our opj.
- **C2 — Swift hybrid decoder module.** From J2K bytes → RGBA `MTLTexture` (reuse the Phase A
  kernels + our opj). Validate the texture matches libdcp's `.image(RGBA)` output (off-screen pixel
  compare) on sample frames — color correctness gate.
- **C3 — wire into the pipeline.** `FrameQueue` carries pooled `MTLTexture` (not `[UInt8]`);
  `MetalVideoView` draws a supplied texture (drop the `upload` byte-copy); decode thread overlaps
  CPU T1 ‖ GPU as in Phase A; honor the parked-on-seek contract. Play a real DCP; **measure sustained
  fps on the M1** (the number that matters).
- **C4 — finish.** `reduce` (proxy resolution factor → `opj_set_decoded_resolution_factor`; the engine
  already has `setDecodeReduction`), 3D/eyes (content is StereoVF; 2D playback picks one eye), and
  GPU color accuracy vs the display colorspace tagging in `MetalVideoView`.

## Open questions / risks

- **opj symbol clash** (C0) — must resolve before anything else.
- **Color accuracy** — GPU xyz_to_rgb must match libdcp's pipeline (gamma 2.6 decode, XYZ→target
  matrix, display encode) within tolerance; validate in C2.
- **Margin under real load** — the 7 ms must absorb distinct-frame decode (vs the harness's warm
  same-frame loop), GPU color, texture/display, and A/V sync. First true reading comes in C3.
- **Non-DCP content** — the player also handles MPEG2/FFmpeg sources; the hybrid path is DCP-J2K only,
  everything else falls back to the existing `.image()` path.
- **4K / HFR** remain out of reach (≈4× both sides) — 2K@24 is the committed target.
