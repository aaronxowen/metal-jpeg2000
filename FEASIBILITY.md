# Feasibility Study: GPU-Accelerated JPEG 2000 on Apple Silicon (Metal)

**Status:** Feasibility & correctness proven · GPU pipeline built and validated (M5 Pro + M1) · architecture decided · remaining work is integration engineering.
**Date:** 2026-06-25
**Target:** GPU-accelerated JPEG 2000 **decode** for Apple Metal on Apple Silicon — concretely, to make realtime DCP playback possible on the low-floor **M1 Mac mini** (the current CPU/OpenJPEG path cannot play even a 2K DCP in realtime).
**Base:** Fork of [OpenJPEG](https://github.com/uclouvain/openjpeg) (UCLouvain, BSD-2), the ISO/ITU JPEG 2000 reference software. `upstream` → `uclouvain/openjpeg`. Prototype code in `proto/`; branch `experiment/metal-t1-gpu`.

---

## Executive summary

**Scope (locked):** decode of existing standards-compliant **IOP + SMPTE DCPs** → Part-1, 9/7,
MQ-coded, DCI 2K/4K cinema profiles. HTJ2K (Part 15) has no DCI compliance route today, so it is out.

**What was built and proven** (methodology: instrument OpenJPEG to dump per-stage corpora + oracles →
portable-C reference → Metal kernel → exact validation):
- The full GPU decode pipeline **minus Tier-2**: Tier-1 (MQ/EBCOT) **bit-exact**, and inverse 9/7 DWT
  + inverse ICT + DC level-shift **integer-exact** vs OpenJPEG — validated on **both the M5 Pro and
  the M1 GPU** (no cross-GPU divergence).

**Key measured facts (M1 Mac mini, real DCI 2K frame):**
- OpenJPEG CPU decode ≈ **57 ms/frame** (all cores) → misses the **41.6 ms/24fps** budget (root problem).
- Tier-1 (MQ coder) ≈ **63%** of decode → CPU T1 ≈ **36 ms**; GPU T1 = **54 ms**; GPU back-end = **37 ms**.

**Architecture decision — hybrid:** **CPU runs T2+T1; the GPU runs the back-end (iDWT+ICT+level-shift),
overlapped across frames** (double-buffered, zero-copy via unified memory). **GPU T1 is abandoned** —
the serial MQ coder runs faster on the CPU (36 ms) than on the GPU (54 ms). The GPU's job is the
back-end and *freeing CPU cores*, not the bottleneck itself. This reframes the original "GPU-accelerate
JPEG 2000" instinct: the win is load-balancing across two processors, not raw GPU throughput.

**Realtime verdict:**
- **2K@24 on the M1: ACHIEVED (empirically), with comfortable margin.** The Phase A overlapped hybrid
  harness measures **34.6 ms/frame (28.9 fps), integer-exact**, on the M1 after a first back-end
  optimization (component-batching) — **7.0 ms margin** over the 41.6 ms budget (was 0.3 ms before opt).
  Still GPU-bound (back-end 32.5 ms vs CPU T1 19 ms). See §17–18.
- **4K / HFR: out of reach** this way (≈4× the work on both sides).
- Guaranteed fallback: `reduce=1` decode (skip finest DWT level, ~75% less T1 work) with quality cost.

**Next step:** optimize the GPU back-end for margin (threadgroup memory, fused gather/scatter, batch
components), then player integration (Phase C). Note: since the M1 is now *GPU-bound*, do **not** add
xyz_to_rgb to the GPU until the back-end is optimized — the CPU has ~19 ms of slack to absorb color.

**Reading guide:** §1–8 initial framing & GPU-mappability · §9–10 DCI scope + in-repo test content ·
§11 CPU profiling (M5 Pro) · §12 GPU Tier-1 (bit-exact) · §13–14 GPU back-end (float/integer-exact) ·
§15–16 **M1 results & the architecture decision** (most current). Sections are chronological; §1's
framing of "Part-1 vs HTJ2K" was resolved in §9.

---

## 1. Initial framing (superseded by the Executive summary)

Building a Metal/Apple-Silicon JPEG 2000 codec is **feasible**, but the value depends almost
entirely on one strategic decision: whether to target classic **Part-1 (MQ arithmetic coder)**
codestreams or **HTJ2K (Part 15, High Throughput)**.

- The **embarrassingly parallel stages** (color transform, wavelet transform, quantization) map
  cleanly to Metal compute and are a guaranteed win.
- The **Tier-1 block coder** is the wall. In classic Part-1 it is the MQ arithmetic coder — serial,
  context-adaptive, feedback-driven — and it is **50–70% of total codec time**. It resists GPU
  parallelism. HTJ2K replaces this stage with a deliberately parallel/SIMD-friendly block coder,
  which is the reason GPU JPEG 2000 implementations (e.g. NVIDIA's nvJPEG2000) post their headline
  throughput numbers on HTJ2K.
- **Apple Silicon's unified memory is a structural advantage** over discrete-GPU CUDA designs: there
  is no PCIe host↔device copy to hide.

---

## 2. The NVIDIA reference: useful, but not as code

The requested reference,
[NVIDIA/CUDALibrarySamples/nvJPEG2000](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/nvJPEG2000),
contains **sample programs only**:

- `nvJPEG2000-Decoder`
- `nvJPEG2000-Decoder-Pipelined`
- `nvJPEG2000-Decoder-Tile-Partial`
- `nvJPEG2000-Encoder`

These call the **precompiled, closed-source `nvjpeg2000` library** shipped in the CUDA Toolkit. The
GPU kernels are not in that repo and are not published anywhere. Consequence:

> nvJPEG2000 is a **design north star and performance benchmark target**, not an implementation
> reference. For actual algorithms we use open code (§6).

One sample is still informative architecturally: `nvJPEG2000-Decoder-Pipelined` exists mainly to
overlap PCIe transfers with compute — a problem Apple Silicon's unified memory **eliminates** (§5).

---

## 3. What this fork gives us

OpenJPEG is the ground-truth reference implementation and our bit-exact validation oracle. Relevant
source (`src/lib/openjp2/`):

| File | Role |
|---|---|
| `mct.c` | Multiple Component (color) Transform — RCT (lossless) / ICT (lossy) |
| `dwt.c` (~150 KB, has SSE/AVX) | Discrete Wavelet Transform — 5/3 reversible, 9/7 irreversible |
| `t1.c` + `mqc.c` | **Part-1 Tier-1**: EBCOT bit-plane coding + MQ arithmetic coder |
| `ht_dec.c` | **HTJ2K (Part 15) block decoder** (no `ht_enc.c` — OpenJPEG only added HT *decode*) |
| `t2.c`, `pi.c` | Tier-2 packetization + packet iterator (codestream assembly) |
| `tcd.c` | Tile coder/decoder orchestration |
| `thread.c`, `bench_dwt.c` | Existing CPU threadpool + DWT benchmark (baseline reference) |

Note: HTJ2K **encode** is not present in OpenJPEG. For that path the reference is OpenJPH/Grok (§6).

---

## 4. Stage-by-stage GPU mappability

| Stage | File(s) | Metal fit | Why |
|---|---|---|---|
| DC level-shift + MCT | `mct.c` | **Excellent** | Per-pixel, embarrassingly parallel |
| DWT (5/3, 9/7) | `dwt.c` | **Excellent** | Separable lifting; well-studied on GPU; the large, easy win |
| Quantization / dequant | `tcd.c`, `t1.c` | **Excellent** | Per-coefficient |
| **Tier-1: EBCOT + MQ (Part 1)** | `t1.c`, `mqc.c` | **Poor** | MQ is a serial, context-adaptive arithmetic coder with per-symbol feedback. Only parallelism is *across* codeblocks (one block/thread → heavy divergence, poor SIMD-lane use). ~50–70% of codec time. |
| **Tier-1: HT block coder (Part 15)** | `ht_dec.c` | **Good** | Purpose-built parallel: MagSgn + MEL + VLC, single cleanup pass instead of 3 sequential bit-plane passes. |
| Tier-2: packetization, PCRD-opt | `t2.c`, `pi.c` | **Poor** | Inherently serial bitstream parsing; normally stays on CPU |

**The decisive fork:**

- **Targeting existing DCI DCPs** → they are Part-1, 9/7, MQ-coded. The hard path. GPU-accelerating
  DWT+MCT+dequant yields real gains, but the serial Tier-1 bounds end-to-end speedup (Amdahl). This
  is why a decade of CUDA Part-1 projects showed only modest end-to-end wins.
- **A new HTJ2K pipeline** → the HT block coder is what makes a GPU codec genuinely worthwhile.
  SMPTE has standardized HTJ2K for IMF; it is the live direction for high-throughput cinema. If the
  codestream is ours to choose, this is the path.

---

## 5. Apple Silicon / Metal specifics

**Advantages (some better than CUDA):**

- **Unified memory is a structural win.** No PCIe, no host↔device copy. `MTLStorageModeShared` gives
  the GPU zero-copy access to codestream and output buffers — removing the entire problem class that
  nvJPEG2000's pipelined decoder is built to hide. Significant for a streaming/real-time cinema decoder.
- High memory bandwidth (M-Max/Ultra ≈ 400–800 GB/s) suits the bandwidth-bound parallel stages.

**Gaps to work through:**

- **No code reuse from CUDA.** MSL is a C++14 dialect; no mature CUDA→Metal transpiler. Kernels are
  rewritten. Concepts port cleanly though:
  - CUDA warp → Metal **SIMD-group** (32-wide on Apple GPUs)
  - `__shfl` / ballot / prefix → `simd_shuffle` / `simd_ballot` / `simd_prefix_*`
  - shared memory → threadgroup memory (~32 KB)
  - `popcount` / `clz` / atomics available for EBCOT bit-twiddling
- **Thousands of tiny codeblock dispatches** → use **indirect command buffers** + `MTLHeap` to avoid
  per-dispatch overhead.
- **Sparse Metal prior art.** Almost all GPU JPEG 2000 work is CUDA/OpenCL; expect to port algorithms,
  not copy Metal code.

---

## 6. References to use (instead of / alongside NVIDIA)

- **OpenJPEG** (this repo) — ground truth + bit-exact validation oracle.
- **OpenJPH** (Aous Naman) — cleanest HTJ2K reference *with SIMD*, by HTJ2K's designer. Best model for
  the HT block coder and the HT *encoder* OpenJPEG lacks.
- **Grok** — modern C++ JPEG 2000 (Part-1 + HT, encode + decode).
- Open CUDA prior art for parallelization patterns: **CUJ2K**, **GPU-JPEG2000**, and academic
  Tier-1-on-GPU papers (Matela et al.).
- **nvJPEG2000 samples** — keep as API shape + throughput yardstick only.

---

## 7. Recommended phased approach (de-risk the wall early)

1. **Baseline & profile** OpenJPEG CPU on the target M-series to quantify the real per-stage split for
   *our* content (2K vs 4K, lossless vs 9/7).
2. **Metal DWT + MCT + dequant** with unified-memory zero-copy; validate bit-exact vs OpenJPEG.
   Low risk, real speedup, proves the harness.
3. **Decide the fork:** Part-1 MQ (compatibility) vs HTJ2K (throughput). Prototype a single HT
   codeblock decode kernel from `ht_dec.c` / OpenJPH and measure the actual GPU win **before**
   committing.
4. Only then attempt full Tier-1 on GPU.

---

## 8. Open questions

- ~~Primary use case: decode existing DCI DCPs (Part-1, forced) vs new HTJ2K pipeline?~~
  **Resolved:** decode existing IOP + SMPTE DCPs → Part-1 / 9-7 / MQ. See §9.
- Real-time playback decode, batch decode, or both? (Sets the per-frame time budget: ~41 ms @ 24 fps,
  ~20 ms @ 48 fps HFR.)
- 2K and/or 4K? (4K ≈ 4× the codeblock count and a 6th DWT level.)
- Minimum supported Apple GPU family (affects SIMD-group features, threadgroup memory).

---

## 9. Decision: decode of DCI Part-1 DCPs — implications

HTJ2K is not yet permitted by the DCI Digital Cinema System Specification, so compliant IOP and
SMPTE DCPs are all Part-1 / 9-7 / MQ-coded. We are on the classic Tier-1 path — but scoping to the
**DCI cinema profiles** (ISO/IEC 15444-1 cinema profiles, carried by SMPTE ST 429-4) removes most of
what makes general JPEG 2000 GPU-hostile.

### What the DCI profiles let us drop

| Parameter | General J2K | **DCI DCP (locked)** | GPU consequence |
|---|---|---|---|
| Wavelet | 5/3 or 9/7 | **9/7 irreversible only** | One DWT kernel path, not two |
| Codeblocks | 16×16 … 64×64, vary at edges | **32×32, uniform** | **Removes the biggest source of thread divergence** — uniform geometry + memory layout |
| Tiling | many tiles | **single tile per image** | No tile orchestration; parallelism is intra-frame codeblocks |
| Component (MCT) transform | optional RCT/ICT | **may be ON** — see note | Conditional inverse ICT stage required; do **not** assume it is absent |
| Resolution levels | variable | **NL=5 (2K), NL=6 (4K)** | Fixed inverse-DWT depth |
| Progression | any | **CPRL** | Predictable precinct/packet walk for Tier-2 |

The uniform **32×32 codeblock** is the key line: the classic "one block per thread → catastrophic
divergence" problem is mostly about *variable* block sizes; under DCI it reduces to content-dependent
divergence only (bitplane count / early termination), not geometry.

**MCT correction (verified against test content):** the in-repo test frames
(`tests/test-content/jpeg200_easyDCP_encoded/`, Rec709 color bars from easyDCP) carry the 2K DCI
profile (`Rsiz=0x0003`) **with MCT enabled (`COD` MCT flag = 1)**. Real X′Y′Z′ distribution DCPs are
typically authored MCT-off, but the DCI 2K profile does not forbid MCT. The decoder must read the COD
flag per codestream and conditionally run the inverse ICT (a trivially parallel per-pixel 3×3 matrix
op — folds into the GPU back-end after the inverse DWT). It cannot be assumed away.

> Confirm precinct sizes and profile-version edge cases against the DCI DCSS and SMPTE ST 429-4 before
> locking kernel assumptions. Verified parameters from the in-repo test frames are in §10.

### The DCI decode pipeline

1. **Decrypt** (AES-128-CBC per KDM) — CPU, hardware AES, not a bottleneck.
2. **Tier-2**: parse codestream → packets per precinct/codeblock. Serial → **CPU**.
3. **Tier-1**: MQ arithmetic decode + EBCOT bitplane passes per 32×32 block. **The wall.**
4. **Dequantize** (9/7 scalar).
5. **Inverse 9/7 DWT** (5 or 6 levels).
6. **Inverse MCT/ICT** — *conditional* on the COD MCT flag (the in-repo test frames have it ON).
7. **Inverse DC level shift** → 12-bit component output.

Block-count sanity check: 2K frame ≈ ~6,000 codeblocks (×3 components), 4K ≈ ~24,000. Ample parallel
work to saturate an Apple GPU across blocks; the constraint is per-block serial latency + divergence
inside Tier-1, never occupancy.

### Recommended architecture: hybrid (CPU Tier-1 + GPU back-end)

OpenJPEG already parallelizes Tier-1 across codeblocks via its CPU threadpool (`thread.c`), and an
M-series has 8–12 fast P-cores that are genuinely good at serial MQ decoding. An all-GPU MQ decoder
must beat that on exactly the stage GPUs hate. So:

- **CPU**: Tier-2 + Tier-1 MQ decode, across all P-cores (lean on OpenJPEG's existing threadpool).
- **GPU**: dequant + inverse 9/7 DWT + level-shift, reading the CPU's coefficient output **zero-copy**
  via `MTLStorageModeShared` — unified memory makes the CPU→GPU handoff free.

Full-GPU Tier-1 becomes a **research track**, pursued only if profiling shows Tier-1 still dominates
after the DWT moves to the GPU. Prototype one 32×32 MQ codeblock kernel and measure before committing.

This makes the safe win (DWT/dequant on GPU, no copy) independent of the risky bet (MQ on GPU).

---

## 10. Verified test content (in-repo)

`tests/test-content/`:
- `jpeg200_easyDCP_encoded/` — 48 `.j2c` frames, color bars (Rec709), easyDCP-encoded.
- `tiffs-adobe-premiere-encoded/` — 48 `.tif` frames (decoded references / source).

Codestream parameters decoded from `Bars+Tone_Rec709_00.j2c` (SIZ + COD markers), representative of
the set (all 48 are identical 41,457-byte frames):

| Field | Value |
|---|---|
| Profile (`Rsiz`) | `0x0003` — 2K Digital Cinema |
| Image size | 1998 × 1080 (DCI 2K Flat) |
| Components | 3 × 12-bit, no subsampling |
| Tiling | single tile (1998 × 1080) |
| Progression | CPRL |
| Quality layers | 1 |
| **MCT** | **enabled (ICT)** |
| Decomposition levels | 5 |
| Codeblock size | 32 × 32 |
| Wavelet | 9/7 irreversible |
| Precincts | 128×128 (lowest res), 256×256 (rest) |

These are a valid first profiling + bit-exact validation target: decode with OpenJPEG CPU, compare to
the matching TIFFs, then use the same frames to validate Metal kernels stage-by-stage.

---

## 11. Profiling results (M5 Pro, 2026-06-25)

**Machine:** Apple M5 Pro, 6 P-cores + 12 E-cores. OpenJPEG 2.5.4, Release, built locally.
**Content:** `Bars+Tone_Rec709_00.j2c` (1998×1080, the §10 frame).

### Wall-clock decode (per frame, `opj_decompress -threads`)

| Threads | Decode time |
|---|---|
| 1 | ~24 ms |
| 6 | ~55 ms (threadpool overhead exceeds gain on a single-tile frame) |
| ALL (18) | ~17 ms |

**Single-threaded 2K decode (~24 ms) is already inside the 24fps budget (41.6 ms).** The GPU value
proposition therefore lives at **4K, HFR (48/60 fps), and multi-stream**, not 2K@24.

### Per-stage attribution (sampling profiler, single-threaded leaf/self-time)

Profiled on two content types — color bars (`jpeg200_easyDCP_encoded/`) and detailed real content
(`jpeg2k-testImage/`, 2K-Flat ProRes422 source, ~228 KB/frame vs 41 KB for bars):

| Stage | Color bars | **ProRes (real)** | GPU fit |
|---|---:|---:|---|
| Tier-1 (MQ / EBCOT) | 36.9% | **71.0%** | Poor (the wall) |
| Inverse DWT (9/7) | 39.6% | 17.8% | Excellent |
| Level-shift / tile→image copy | 20.4% | 9.1% | Excellent (per-pixel) |
| Inverse MCT / ICT | 2.3% | 1.1% | Excellent (per-pixel) |
| Tier-2 (packet parse) | 0.8% | 1.1% | Serial → CPU |

Wall-clock, ProRes content: **53 ms single-thread / 35 ms all-cores** (vs 24 / 17 ms for bars).

### Interpretation — content type flips the strategy

- **Real content is Tier-1-bound (~71%), textbook 50–70%.** The color bars (T1 ~37%) were a misleading
  best case — highly compressible → few coding passes. Detailed content confirms the classic reality:
  the MQ coder is the bottleneck. **Use the ProRes numbers, not the bars, for all sizing.**
- **The GPU back-end (DWT + level-shift + MCT) addresses only ~28% of real-content decode time.**
  Amdahl ceiling if that 28% becomes free: **~1.4× (53 → ~38 ms at 2K)**. And CPU all-cores already
  does 2K real content in 35 ms, so the back-end alone buys little at 2K.
- **At 4K (~4× work) the Tier-1 wall dominates and the back-end-only hybrid cannot reach real-time**
  (~150–210 ms/frame; shaving 28% leaves ~110–150 ms, 3–4× over the 41.6 ms budget).
- **OpenJPEG's DWT is already NEON-vectorized** (`opj_v8dwt_interleave_h`); the Metal DWT competes
  against SIMD'd CPU code. The win comes from GPU bandwidth/parallelism at 4K, not out-vectorizing at 2K.

### Revised conclusion (supersedes the §9 "optional research track" framing)

For the committed use case (real DCP content, especially 4K / HFR), **GPU Tier-1 is on the critical
path, not optional.** The back-end-first hybrid is still the right *first* deliverable (low risk,
de-risks the harness, helps 2K), but it caps at ~1.4×. Meaningful speedup at 4K **requires** a Metal
MQ-decoder kernel: one 32×32 codeblock per thread, parallel across the ~6,000 blocks/frame.

Prior-art reality check: this is where open CUDA Part-1 implementations earned **modest ~2–4× wins**,
not the 10–50× of embarrassingly parallel work — and why nvJPEG2000's headline throughput is all
HTJ2K, never Part-1 MQ. GPU Tier-1 for Part-1 is feasible but is the project's principal risk; it
should be prototyped and measured (single-codeblock kernel) before committing to a full build.

### Reproduce

```sh
# Build (CMake 4.x ok; OpenJPEG uses VERSION 3.10...3.31.5 range syntax):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j

# Verify codestream params:
DYLD_LIBRARY_PATH=build/bin build/bin/opj_dump -i <frame>.j2c

# Time a decode:
DYLD_LIBRARY_PATH=build/bin build/bin/opj_decompress -i <frame>.j2c -o /tmp/o.tif -threads ALL

# Per-stage profile: in-process decode loop, profiled under macOS `sample`:
clang -O2 -Isrc/lib/openjp2 -Ibuild/src/lib/openjp2 tests/bench_decode.c -o /tmp/bench_decode \
  -Lbuild/bin -lopenjp2 -Wl,-rpath,$(pwd)/build/bin
/tmp/bench_decode <frame>.j2c 600 1 & sample $! 6 -file /tmp/oj_sample.txt
```

---

## 12. T1 GPU kernel prototype — progress

**Goal (decided):** retire the principal risk by porting one 32×32 MQ-decode codeblock kernel to
Metal, benchmark vs OpenJPEG CPU (`opj_t1_clbl_decode_processor`), validate bit-exact.

### Corpus + oracle (done)

`opj_t1_decode_cblk` (src/lib/openjp2/t1.c) has an **env-gated corpus dump** (`OPJ_T1_DUMP=<path>`,
decode with `-threads 1`). Per non-trivial codeblock it writes: `orient, roishift, cblksty, numbps,
w, h, num_segs`, per-seg `(len, real_num_passes)`, total compressed bytes + the bytes, then the
decoded coefficients (w·h int32) as a **bit-exact oracle**. Extract:

```sh
OPJ_T1_DUMP=proto/corpus.bin DYLD_LIBRARY_PATH=build/bin \
  build/bin/opj_decompress -i <frame>.j2c -o /tmp/x.tif -threads 1
```

### Workload characterization (one ProRes 2K frame → `proto/corpus.bin`, 11.6 MB)

- **2,972 non-trivial codeblocks** (empty/zero blocks return before decode → excluded; these are the
  real T1 work). Corpus parses to exact EOF.
- **All cblksty=0, all single-segment** → confirms the kernel needs only the MQ path of the three
  passes: no RAW/bypass, no RESTART, no vertical-stripe-causal, no segment-symbol. Major simplification.
- **84% are full 32×32**; remainder are edge blocks (smaller w/h ≤ 32). Kernel handles variable w,h.
- **orient** mix HL/LH/HH (1/2/3) dominant, few LL — selects the ZC context LUT.
- **numbps 1–13** (broad) → the per-block divergence source (variable pass count).
- Compressed bytes/block: min 2, max 1520, mean 74 (≈221 KB total, matches frame payload).

### Reference decoder (done — bit-exact)

`proto/t1_ref.c` — a self-contained, dependency-free port of the MQ coder + all three decode passes
(sigpass/refpass/clnpass), the cblksty=0 MQ-only path. Reads `proto/corpus.bin`, decodes every block,
compares to the oracle:

```
codeblocks: 2972   PASS: 2972   FAIL: 0
CPU reference T1 decode: 41.82 ms for 2972 cblks (14.1 us/cblk) [plain C, single-thread, M5 Pro]
```

**All 2,972 blocks bit-exact.** This validates the full algorithm understanding + corpus completeness,
and is the exact code that ports to MSL (written with plain arrays/indices, no library pointers).
Build/run: `clang -O2 proto/t1_ref.c -o /tmp/t1_ref && /tmp/t1_ref proto/corpus.bin`.

### Metal kernel + benchmark (done — bit-exact, working)

`proto/t1_kernel.metal` (1 thread = 1 codeblock, near-mechanical port of `t1_ref.c`),
`proto/t1_luts.metal` (generated from `t1_luts.h`), `proto/t1_bench.swift` (headless harness:
unified-memory buffers, dispatch, bit-exact check, timing).

```
parsed 2972 codeblocks ... GPU: Apple M5 Pro
BIT-EXACT: all 2821292 coeffs match oracle ✓
GPU T1 decode: best 14.684 ms for 2972 cblks (4.94 us/cblk)
```

| Decoder (M5 Pro) | Time / frame | Per block | |
|---|---|---|---|
| CPU reference (plain C, 1 thread) | 41.8 ms | 14.1 µs | scalar port |
| **GPU Metal kernel** | **14.7 ms** | **4.9 µs** | **~2.85× vs scalar CPU, bit-exact** |

Build/run:
```sh
python3 proto/gen_luts.py    # (the snippet that wrote proto/t1_luts.metal)
swiftc -O proto/t1_bench.swift -o /tmp/t1_bench -framework Metal -framework Foundation
/tmp/t1_bench proto/corpus.bin proto/t1_luts.metal proto/t1_kernel.metal
```

### Interpretation + headroom

- **The principal risk is retired: GPU Part-1 T1 decode is feasible and bit-exact**, at ~2.85× the
  equivalent scalar CPU code — squarely in the prior-art band (~2–4× for Part-1 MQ; the big numbers
  are HTJ2K-only). It does **not** 10× the way embarrassingly parallel stages do; the MQ coder's
  serial nature and per-block divergence are the limiters, as expected.
- This kernel is **unoptimized**: flags/data live in device memory (uncoalesced), no threadgroup
  tiling, and a single frame is only 2,972 threads — too few to saturate the M5 Pro GPU, so the time
  is divergence/latency-bound (the numbps=13 blocks gate the dispatch). Headroom: threadgroup-memory
  scratch, batching several frames per dispatch, and packing work by block size to cut divergence.
- **Caveat — measured on M5 Pro, not the M1 target.** The whole toolchain is built to run on the M1:
  the relative GPU-vs-CPU ratio there is the real question. On the smaller M1 GPU, 2,972 threads
  fills the machine *better* (relatively), and the M1 CPU is weaker, so the GPU's relative advantage
  may be larger — but this must be measured. Run `/tmp/t1_bench` on the M1.

### Bottom line for the project

For 2K real content the hybrid (CPU keeps a tuned multithreaded T1) is already close; GPU T1 buys a
modest multiple and, more importantly on the M1, **moves the bottleneck stage off the 4 starved
P-cores onto the idle GPU** while the back-end (DWT/MCT/level-shift/xyz_to_rgb) also offloads. The
realistic path to M1 2K realtime: GPU back-end + GPU color first (large, easy), then GPU T1 with
threadgroup-memory optimization if profiling on the M1 still shows a gap. 4K/HFR will need the
optimized GPU T1.

---

## 13. GPU back-end: inverse 9/7 DWT (done — float-exact)

Same methodology as T1. Corpus dump added to `opj_dwt_decode_real` (`OPJ_DWT_DUMP=<path>`, `-threads 1`):
per tile-component it emits resolution geometry + input float buffer + output float buffer (oracle).

- `proto/dwt_ref.c` — scalar port of `opj_dwt_decode_tile_97` / `opj_v8dwt_decode` (NB_ELTS=1), the
  6-step lifting incl. the deliberate `two_invK` quirk. Build with `-ffp-contract=off` (matches the
  non-fused NEON `vmlaq_f32`). Max abs diff vs oracle 0.0045 (FP-order noise; rounds to same int image).
- `proto/dwt_kernel.metal` + `proto/dwt_bench.swift` — Metal kernel: 1 thread per row (H pass) / per
  column (V pass); barrier between H/V and between levels (separate encoders in one command buffer).
  Compiled with fast-math **off** → GPU output is **float-exact** vs the oracle (max abs diff 0.0).

```
GPU: Apple M5 Pro  ... 3 components, 6 res levels each
records: 3  coeffs: 6473520  differ: 0  max abs diff: 0.0   <- FLOAT-EXACT
GPU iDWT: 12.65 ms/frame (4.22 ms/component)
```

| Inverse 9/7 DWT (frame, 3 comp) | Time | |
|---|---|---|
| scalar C reference | 22.4 ms | no SIMD |
| GPU Metal kernel | 12.7 ms | float-exact, **unoptimized** |
| OpenJPEG CPU (NEON, ≈18% of 53 ms) | ~9.4 ms | already vectorized |

### Key finding — the DWT is NOT a clear GPU win on a strong-CPU Mac

Unlike T1, the inverse DWT is memory-bandwidth-bound and OpenJPEG's CPU path is already
NEON-vectorized, so the unoptimized GPU kernel (per-thread device-memory scratch, uncoalesced
gather/scatter, 10 dispatches/frame) **loses to the M5 Pro CPU**. Implications:

- The DWT's GPU value is **machine-dependent**: it pays off on the weak-CPU **M1** and, more
  importantly, by **freeing the 4 P-cores for T1** — not as a raw speedup on strong Macs.
- Kernel headroom: threadgroup-memory tiling, fuse gather/scatter (avoid the separate interleaved
  buffer), batch all 3 components into one dispatch, fewer/larger dispatches.
- **Re-prioritization:** the bigger *easy* CPU-relief win is likely **`xyz_to_rgb`** (libdcp does it
  on CPU, per-pixel `pow`, **not** NEON-optimized) — a better next target than further DWT tuning.

### Correctness milestone

Both GPU stages built so far — **T1 (bit-exact) and inverse DWT (float-exact)** — validate against
OpenJPEG. The methodology (corpus + oracle + scalar reference + Metal kernel) is proven and reusable
for the remaining back-end stages (inverse MCT/ICT, level-shift, xyz_to_rgb).

---

## 14. GPU back-end complete: iDWT + inverse ICT + level-shift (integer-exact)

Frame-level corpus dump added to `opj_tcd_decode_tile` (`OPJ_BACKEND_DUMP=<path>`, `-threads 1`):
per frame it emits geometry + per-component (prec, sgnd, dc_level_shift) + mct flag + the 3 post-T1
input float buffers, then (after the full back-end) the 3 final integer component buffers (oracle).

- `proto/backend_ref.c` — scalar reference for the whole back-end (iDWT + inverse ICT
  `r=y+1.402v / g=y−0.34413u−0.71414v / b=y+1.772u` + `clamp(lrintf(v)+dc, lo, hi)`).
  Integer-exact on 6,470,670 / 6,473,520 samples; 2,850 off-by-1 (scalar iDWT FP-order noise).
- `proto/backend_kernel.metal` + `proto/backend_bench.swift` — reuses the iDWT H/V kernels, adds a
  `backend_finalize` kernel (1 thread/pixel: inverse ICT + level-shift + clamp → int). fast-math off.

```
GPU: Apple M5 Pro  frame 1998x1080, 3 comps, 6 res, mct=1
samples: 6473520  differ: 0  max |diff|: 0    <- INTEGER-EXACT vs OpenJPEG final image
GPU full back-end (iDWT+ICT+level-shift): 14.0 ms/frame
```

### Pipeline status — the whole DCI decode minus Tier-2 is now GPU + exact

| Stage | GPU status | M5 Pro time/frame |
|---|---|---|
| Tier-2 (packet parse) | CPU (serial, stays) | small |
| **Tier-1 (MQ/EBCOT)** | **bit-exact** | 14.7 ms (unopt) |
| **iDWT + ICT + level-shift** | **integer-exact** | 14.0 ms (unopt) |
| xyz_to_rgb (display) | not yet (libdcp CPU stage) | — |

Both GPU halves of the codec now reproduce OpenJPEG output exactly. Remaining work is **optimization**
(threadgroup memory, batching, fused stages) and **measuring on the M1**, plus the separate
`xyz_to_rgb` color stage (the likely highest-yield easy CPU-relief win, per §13).

---

## 15. M1 results + analysis (the real target)

Run on the actual M1 Mac mini (4 P-cores, ~8-core GPU) via `proto/M1_RUNBOOK.md`; raw numbers in
`proto/M1_RESULTS.md`. Corpus oracle held on the M1 GPU: **T1 bit-exact, back-end integer-exact** —
no cross-GPU divergence (key portability win).

| Stage | M1 GPU | vs scalar CPU ref |
|---|---|---|
| T1 (MQ/EBCOT) | 54.3 ms/frame | 1.44× |
| iDWT | 35.9 ms/frame | 1.19× |
| Full back-end (iDWT+ICT+shift) | 37.0 ms/frame | 1.10× |

### Honest interpretation (corrects the M1_RESULTS verdict)

- **Not realtime yet.** Unoptimized, T1 (54 ms) alone exceeds the 41.6 ms/24fps 2K budget; T1 +
  back-end sequentially ≈ 91 ms ≈ **2.2× over budget**.
- **The 1.1–1.44× margins overstate the advantage.** The CPU ref is clean-room scalar single-thread,
  NOT OpenJPEG's NEON + 4-core threadpool. These ratios do **not** establish that the GPU beats the
  real CPU decoder.
- **Pipelining T1+back-end across frames does NOT help** as proposed: both run on the one GPU, so the
  problem is throughput-bound (≈91 ms of GPU work/frame → ~11 fps ceiling), not latency-bound.
  Pipelining only adds throughput when **different processors** run different stages — i.e. the
  **hybrid: CPU does T1 while the GPU does the previous frame's back-end**, giving per-frame
  `max(CPU_T1, GPU_backend≈37 ms)` instead of the sum.

### Critical missing measurement (next step)

Everything hinges on **OpenJPEG's actual M1 decode time — NEON, `-threads ALL` — total and T1-only.**
That determines: (a) the true baseline gap (the current failing path), (b) whether GPU T1 (54 ms)
beats CPU T1 or T1 should stay on the CPU, and (c) whether `max(CPU_T1, 37 ms GPU back-end)` clears
41.6 ms. Build OpenJPEG on the M1 and time `opj_decompress` + the per-stage `sample` profile (as in §11).

### Decision tree once that number exists

- If **CPU_T1 (NEON, 4-thread) < ~41 ms**: hybrid wins — keep T1 on CPU, move back-end to GPU,
  overlap across frames → likely realtime 2K. Lowest risk.
- If **CPU_T1 ≥ ~41 ms**: T1 itself must get faster. Optimize the GPU T1 kernel (threadgroup-memory
  flags/data, cut divergence, batch) targeting ~2× → ~27 ms, then hybrid or all-GPU-optimized.
- Fallback for guaranteed realtime: decode at `reduce=1` (skip finest DWT level, ~75% less T1 work)
  with quality tradeoff.

---

## 16. M1 Phase-2 baseline + architecture decision

OpenJPEG real decode on the M1 (NEON, threadpool), DCI 2K frame:

| | Time | |
|---|---|---|
| `opj_decompress -threads 1` | 106 ms/frame | |
| `opj_decompress -threads ALL` | **57 ms/frame** | confirms: OpenJPEG misses the 41.6 ms budget → the root problem |
| T1 share (sampled) | **63.3%** | DWT ~18.6%, MCT ~1.4%, other (T2/alloc/IO) ~16.7% |
| → CPU T1, all cores | **≈ 36 ms** | 57 × 63.3% |

### Decision: hybrid (CPU T1 ‖ GPU back-end). GPU T1 abandoned.

GPU T1 (54 ms) is **slower** than 4-core CPU T1 (36 ms) → don't put T1 on the GPU. Correct
architecture: **CPU runs T2+T1; GPU runs the back-end (iDWT+ICT+level-shift); overlap across frames**
(double-buffered, zero-copy via unified memory). Steady-state ≈ `max(CPU side, GPU side)`.

### Correction to the M1_RESULTS verdict — it's tight, not a clean win

That verdict used `max(CPU_T1=36, GPU_backend=37) ≈ 37 ms ✓`, but the CPU side keeps **everything
except the back-end** (~80% of 57 ms ≈ **45.6 ms**), not just T1. So the real critical path is
`max(45.6, 37) ≈ 45.6 ms` — slightly **over** the 41.6 ms budget as measured.

Caveats that pull it back under:
- Much of "other ~16.7%" is alloc/memset/IO — partly a per-iteration harness artifact; **buffer reuse**
  in a real player removes most, pulling the CPU side toward ~37–40 ms. T2 itself is ~1%.
- "-threads ALL" was reported as **4 P-cores**; the M1 also has 4 E-cores — possibly untapped for T1.

**Honest conclusion:** realtime 2K@24 on the M1 via the hybrid is **plausible and close (~37–46 ms vs
41.6 ms) but not yet proven** — both sides balance right at the budget line. 4K/HFR won't fit this way.

### Next: prove it with an integration prototype

The decisive test is no longer a kernel microbenchmark — it's an **end-to-end overlapped hybrid**:
CPU T1 (OpenJPEG, buffers reused) for frame N+1 running concurrently with the GPU back-end for frame N,
measured as sustained fps on the M1. Secondary levers if it lands just over: optimize the GPU back-end
(threadgroup memory — frees scheduling slack), use M1 E-cores for T1, or `reduce=1` fallback.

---

## 17. Phase A decision gate — RESULT: realtime 2K on M1 achieved

The overlapped hybrid harness (`proto/phase_a/hybrid_harness.swift`: CPU decode-to-T1 via the fork's
`opj_set_t1_output_callback` ‖ GPU back-end, 2-slot ping-pong, shared buffers) was run on the M1 over
120 frames of real DCI 2K content.

| Metric (M1) | Value |
|---|---|
| **Overlapped** | **41.3 ms/frame · 24.2 fps** |
| CPU decode-to-T1 (calibration) | 19.0 ms/frame |
| GPU back-end (calibration) | 38.5 ms/frame |
| Critical path `max(CPU,GPU)` | 38.5 ms — **GPU-bound** |
| Validation | **integer-exact vs oracle** |
| 41.6 ms / 24fps budget | **MET ✓** (0.3 ms margin) |

(M5 Pro reference: ~20 ms/frame, ~49 fps.)

### Reconciliation with §16 (the bottleneck flipped — for the better)
§16 derived CPU T1 ≈ 36 ms and feared the CPU side (T1+T2+overhead ≈ 45 ms) would bind. Measured
decode-to-T1 is **19 ms** — much lower. Why: the decode-to-T1 path **skips the tile→image copy** (and
the back-end), uses all 8 cores incl. E-cores, and the harness decodes the same frame repeatedly
(warm caches). So the CPU side is far more comfortable than estimated, and the **GPU back-end is now
the binding stage**.

### Robustness of the result
The harness decodes one frame 120× (warm), which may flatter the CPU side vs real playback of distinct
frames. But the CPU has ~19 ms of slack under the GPU's 38.5 ms, so even a meaningfully higher real-
playback CPU T1 stays hidden behind the GPU — the 24fps result is robust. The GPU side is data-content
independent and unaffected.

### Implications for next steps
- **The binding constraint is the *unoptimized* GPU back-end.** Optimizing it (threadgroup-memory
  scratch, fuse the iDWT gather/scatter, batch the 3 components, fewer dispatches) directly widens the
  thin 0.3 ms margin — likely to a comfortable one.
- **Do not move xyz_to_rgb onto the GPU yet** (Phase B): the M1 is GPU-bound, so adding GPU work hurts.
  The CPU has slack — keep color on the CPU, or optimize the back-end first, then reconsider.
- 2K@24 is the target and it passes; 4K/HFR remain out of reach (≈4× both sides).
- The core feasibility question is now answered **empirically and affirmatively**. Remaining work is
  back-end kernel optimization (for margin) and player integration (Phase C).

---

## 18. Phase A back-end optimization #1 (component-batching) — M1 result

First post-decision-gate optimization: concatenate the 3 components and batch them into single
inverse-DWT H/V dispatches (`proto/phase_a/backend_opt.metal`) — ~30→~10 dispatches, 3× threads per
dispatch. Per-line lifting math unchanged → still **integer-exact**.

| GPU back-end (per frame) | M5 Pro | M1 |
|---|---|---|
| original | 14.1 ms | 38.5 ms |
| component-batched | 8.4 ms (**1.68×**) | 32.5 ms (**1.18×**) |

| Overlapped hybrid (M1) | before | after |
|---|---|---|
| ms/frame · fps | 41.3 · 24.2 | **34.6 · 28.9** |
| margin over 41.6 ms | 0.3 ms | **7.0 ms** |
| validation | exact | **integer-exact** |

### Why the M1 gained less (1.18×) than the M5 Pro (1.68×)
Batching cuts dispatch overhead and raises occupancy, but **not** total device-memory traffic. The M1's
small GPU is **bandwidth-bound**, and the per-thread DWT working buffer (`Wpool`) lives in device
memory — every lifting sub-pass streams it through DRAM. So the M1 win is occupancy-limited. The next
lever, if more margin is wanted, is **threadgroup-memory scratch** for the 1-D line (lifting hits
on-chip memory; only the initial gather + final scatter touch DRAM) — that targets bandwidth directly
and should help the M1 more than batching did. Deferred: 7 ms margin is already comfortable for 2K@24.

### Status
2K@24 realtime on the M1 is met with ~7 ms headroom, integer-exact. The harness still outputs XYZ
(no `xyz_to_rgb` yet); since the pipeline is GPU-bound with CPU T1 idle ~14 ms/frame, color belongs on
the **CPU slack**, not the GPU. Remaining margin must also absorb real-player overheads (distinct-frame
decode vs the harness's warm same-frame loop, MXF read + AES decrypt, display, A/V sync) — to be
measured in Phase C. Recommendation: stop micro-optimizing the back-end now and proceed to Phase C
(player integration), optimizing in situ against real overheads.
