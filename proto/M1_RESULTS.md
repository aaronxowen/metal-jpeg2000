# M1 Benchmark Results

**Date:** 2026-06-25  
**Hardware:** Apple M1 (4 P-cores), Apple M1 GPU (~8 GPU cores)  
**Branch:** experiment/metal-t1-gpu

---

## T1 (MQ/EBCOT decode)

| | Time | µs/cblk |
|---|---|---|
| GPU | 54.331 ms/frame (best), 56.134 ms avg | 18.281 µs/cblk |
| CPU scalar ref | 78.41 ms/frame | 26.4 µs/cblk |
| GPU / CPU ratio | **1.44× faster** | |

Corpus: 2,972 codeblocks, 2,821,292 coeffs  
**Correctness: BIT-EXACT ✓** (all 2,821,292 coeffs match oracle)

---

## iDWT (inverse 9/7 DWT)

| | Time |
|---|---|
| GPU | 35.863 ms total (11.954 ms/component) |
| CPU scalar ref | 42.62 ms total (14.21 ms/component) |
| GPU / CPU ratio | **1.19× faster** |

Corpus: 3 components, 1998×1080, 6 res levels, 6,473,520 coeffs  
**Correctness: within tolerance (rounds to same integer image) ✓**

---

## Full back-end (iDWT + ICT + DC level-shift)

| | Time |
|---|---|
| GPU | 36.964 ms/frame |
| CPU scalar ref | 40.81 ms/frame |
| GPU / CPU ratio | **1.10× faster** |

Frame: 1998×1080, 3 comps, 6 res, MCT on, 6,473,520 samples  
**Correctness: INTEGER-EXACT vs OpenJPEG final image ✓**

---

## Verdict

| Stage | GPU wins? | Margin |
|---|---|---|
| T1 | Yes | 1.44× |
| iDWT | Yes | 1.19× |
| Back-end | Yes | 1.10× |

The M1's weaker CPU gives the GPU a clearer T1 win than was seen on the M5 Pro. However, **T1 alone at 54.3 ms already exceeds the 41.6 ms/frame @ 24fps 2K realtime budget**. The back-end (37.0 ms) fits the budget individually, but the combined sequential pipeline (T1 + back-end ≈ 91 ms) is ~2.2× over budget. Pipelining T1 and back-end across frames is the next lever to investigate.

The scalar CPU ref is a clean-room reference, not OpenJPEG's tuned NEON path — it undersells the real CPU floor. The headline result remains GPU throughput on the M1 GPU.

---

## Phase 2 — OpenJPEG Real CPU Baseline

### Total decode time (opj_decompress, real DCI 2K frame)

| Threads | Decode time |
|---|---|
| `-threads 1` | 106 ms/frame |
| `-threads ALL` (4 P-cores) | **57 ms/frame** |

### T1 fraction (sampled from single-threaded bench_decode, 400 iters)

| Symbol group | Samples | Share |
|---|---|---|
| `opj_t1_*` (MQ/EBCOT) | 3,029 / 4,787 | **63.3%** |
| `opj_v8dwt_*` + `opj_dwt_*` (DWT) | ~890 | ~18.6% |
| `opj_mct_*` (MCT/ICT) | 67 | ~1.4% |
| Other (T2 parse, alloc, I/O) | ~801 | ~16.7% |

### Derived real CPU T1 time

| | T1 time |
|---|---|
| Single-thread (1 P-core) | 106 ms × 63.3% ≈ **67 ms** |
| All-cores (4 P-cores) | 57 ms × 63.3% ≈ **36 ms** |

### Architecture verdict

The key question: does `max(CPU_T1, GPU_backend)` fit the 41.6 ms budget?

- **All-cores CPU T1 ≈ 36 ms**, GPU back-end ≈ 37 ms → critical path `max(36, 37)` ≈ **37 ms — within budget ✓**
- The GPU T1 kernel (54 ms) is **slower** than 4-core NEON T1 (36 ms); GPU T1 is not the right path.
- **Recommended architecture: CPU T1 (all 4 cores) pipelined with GPU back-end across frames.**
  This leaves headroom (~4.6 ms) and keeps all CPU cores busy on the codec bottleneck while the GPU handles the back-end in parallel.

---

## Phase A-2 — Overlapped Hybrid Harness (decision-gate)

CPU decodes frame N+1 through Tier-1 (multithreaded) while GPU runs the back-end on frame N.  
Per-frame time = `max(CPU T1, GPU back-end)`. 120-frame run on real DCI 2K content.

| Metric | Value |
|---|---|
| **Overlapped ms/frame** | **41.325 ms** |
| **fps** | **24.2 fps** |
| CPU T1 calibration | 18.998 ms/frame |
| GPU back-end calibration | 38.520 ms/frame |
| Critical path `max(CPU, GPU)` | 38.520 ms (GPU-bound) |
| Validation | **INTEGER-EXACT vs oracle ✓** |
| **MEETS 24fps budget (41.6 ms)?** | **YES ✓** |

The hybrid delivers 2K realtime on the M1. The GPU back-end is the bottleneck; CPU T1 finishes well inside it at 19 ms. Margin over budget is 0.3 ms — thin but passing. (M5 Pro reference: ~20 ms/frame, ~49 fps.)

---

## Phase A optimized back-end (M1) — 2026-06-25

Component-batching optimization applied to the GPU back-end kernel. 120-frame run on real DCI 2K content.

| Metric | Value |
|---|---|
| **Optimized GPU back-end (Step 4)** | **32.502 ms/frame** |
| vs earlier 38.520 ms | **1.18× faster** |
| CPU decode-to-T1 calibration | 19.195 ms/frame |
| GPU back-end calibration (Step 3) | 33.939 ms/frame |
| Critical path `max(CPU, GPU)` | 33.939 ms (GPU-bound) |
| **Overlapped hybrid ms/frame** | **34.608 ms** |
| **fps** | **28.9 fps** |
| Validation | **INTEGER-EXACT vs oracle ✓** |
| **MEETS 24fps budget (41.6 ms)?** | **YES ✓** |
| Margin | **7.0 ms** (vs 0.3 ms before) |

The back-end optimization adds 7.0 ms of margin over the 41.6 ms budget, up from 0.3 ms. The pipeline is now comfortably GPU-bound at 33.9 ms with CPU T1 idle for ~14 ms of each frame interval.

---

## Phase A threadgroup back-end sweep (M1) — 2026-07-07

Threadgroup-memory kernel (`backend_tg.metal`) — one threadgroup per DWT line, lifting scratch in threadgroup memory, no device-memory Wpool. Swept `threadsPerGroup` across 64/128/256/512. All runs INTEGER-EXACT vs oracle ✓.

| threadsPerGroup | ms/frame | vs backend_opt (32.502 ms) |
|---|---|---|
| 64  | 5.174 ms | 6.29× faster |
| 128 | 4.269 ms | 7.62× faster |
| **256** | **4.229 ms** | **7.69× faster** ← best |
| 512 | 4.637 ms | 7.01× faster |

Best: **T=256, 4.229 ms/frame** — a 7.7× improvement over the previous optimized kernel.

This drops the GPU back-end far below CPU T1 (~19 ms), flipping the bottleneck from GPU-bound to **CPU-bound**. The new theoretical overlapped ceiling is `max(19, 4.2)` ≈ **19 ms/frame (~52 fps)** — well inside the 41.6 ms 2K@24fps budget with ~22 ms of margin. The hybrid harness needs re-running with `backend_tg.metal` to confirm the end-to-end number.
