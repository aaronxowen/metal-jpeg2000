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
