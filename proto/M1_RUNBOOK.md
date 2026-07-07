# M1 benchmark runbook (for a Claude session on the M1 Mac mini)

## Context
`metal-jpeg2000` is a GPU/Metal JPEG 2000 **decode** prototype (fork of OpenJPEG). It exists to fix a
real problem: `swift-dcp-player` cannot achieve realtime playback of even a 2K DCP on **this M1 Mac
mini** (4 P-cores / ~8 GPU cores). The prototype was built and validated on an M5 Pro; **this machine
(the M1) is the real target**, so we need the GPU-vs-CPU numbers here.

Two GPU stages are already built and proven bit/integer-exact vs OpenJPEG on the M5 Pro:
- **Tier-1 (MQ/EBCOT) decode** — the codec bottleneck (~71% of decode time on real content).
- **Back-end** — inverse 9/7 DWT + inverse ICT + DC level-shift (the rest, minus Tier-2 packet parse).

Full study: `FEASIBILITY.md` (§12 T1, §13 DWT, §14 back-end).

## What we need from this run
The M5 Pro has an unusually strong CPU, so there the GPU only modestly beat (T1) or even lost (DWT) to
the CPU. **The M1 has a much weaker CPU and the GPU is otherwise idle** — the question is whether the
GPU clearly wins *here*, and whether the numbers point toward realtime 2K (budget: 41.6 ms/frame @ 24 fps).

## Prerequisites
- macOS with Xcode command-line tools (`swiftc`, `clang`).
- The 3 corpus files present in `proto/`: `corpus.bin`, `dwt_corpus.bin`, `backend_corpus.bin`.
  (They are gitignored derived data; if missing, see "Regenerating corpora" below.)
- No OpenJPEG build is required for the benchmarks — they are standalone.

## Run these (from the repo root)

```sh
# 0. Confirm this is an M1 and note the GPU name.
sysctl -n machdep.cpu.brand_string
system_profiler SPDisplaysDataType | grep -i chipset || true

# --- GPU benchmarks (Metal). fast-math is forced off inside the harnesses. ---
swiftc -O proto/t1_bench.swift      -o /tmp/t1_bench      -framework Metal -framework Foundation
swiftc -O proto/dwt_bench.swift     -o /tmp/dwt_bench     -framework Metal -framework Foundation
swiftc -O proto/backend_bench.swift -o /tmp/backend_bench -framework Metal -framework Foundation

/tmp/t1_bench      proto/corpus.bin         proto/t1_luts.metal proto/t1_kernel.metal
/tmp/dwt_bench     proto/dwt_corpus.bin     proto/dwt_kernel.metal
/tmp/backend_bench proto/backend_corpus.bin proto/backend_kernel.metal

# --- CPU scalar references (for an M1 CPU baseline on the same data) ---
clang -O2                   proto/t1_ref.c      -o /tmp/t1_ref
clang -O2 -ffp-contract=off proto/dwt_ref.c     -o /tmp/dwt_ref
clang -O2 -ffp-contract=off proto/backend_ref.c -o /tmp/backend_ref -lm

/tmp/t1_ref      proto/corpus.bin
/tmp/dwt_ref     proto/dwt_corpus.bin
/tmp/backend_ref proto/backend_corpus.bin
```

## What to report back
For each of T1 / DWT / back-end, report on the M1:
1. **Correctness** — does the GPU bench still print bit-exact / float-exact / integer-exact? (It must.)
2. **GPU time/frame** (and µs/cblk for T1) from the `*_bench` output.
3. **CPU scalar ref time/frame** from the `*_ref` output.
4. **GPU vs CPU-scalar ratio** on the M1.
5. The GPU name and CPU brand string.

Then a one-line verdict per stage: does the GPU win on the M1, and by how much? And does the combined
GPU pipeline (T1 + back-end) look like it could fit the 41.6 ms/frame 24 fps budget for 2K?

Note: the scalar CPU ref is *not* OpenJPEG's tuned NEON path — it's a clean-room reference. It's a fair
"equivalent scalar code" baseline but undersells the real CPU. The headline question is still GPU
throughput on the M1 GPU. (A tuned-CPU comparison would require building OpenJPEG; skip unless asked.)

## Phase 2 — OpenJPEG real-CPU baseline on the M1 (CRITICAL next measurement)

The Phase-1 GPU-vs-CPU ratios use a clean-room *scalar single-thread* CPU ref, which undersells the
real CPU. To decide the architecture we need OpenJPEG's **actual** M1 decode time (NEON + threadpool).
This requires building OpenJPEG and one real DCI 2K `.j2c` frame (transfer one from the M5 box:
`tests/test-content/jpeg2k-testImage/2K-Flat_ProRes422_StereoVF_000000.j2c`, ~228 KB).

```sh
# Build OpenJPEG (needs cmake + libpng/libtiff/lcms2; brew install them, or use Kitware cmake binary).
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j

FRAME=path/to/2K-Flat_ProRes422_StereoVF_000000.j2c
# Total decode time, single-thread and all-cores:
for T in 1 ALL; do echo "threads=$T:"; DYLD_LIBRARY_PATH=build/bin build/bin/opj_decompress -i "$FRAME" -o /tmp/x.tif -threads $T 2>&1 | grep "decode time"; done
```

Optional per-stage split (confirms the T1 fraction on the M1) — compile the in-process loop and sample it:
```sh
clang -O2 -Isrc/lib/openjp2 -Ibuild/src/lib/openjp2 tests/bench_decode.c -o /tmp/bench_decode \
  -Lbuild/bin -lopenjp2 -Wl,-rpath,$(pwd)/build/bin
/tmp/bench_decode "$FRAME" 400 1 & sample $! 6 -file /tmp/m1_sample.txt
awk '/Sort by top of stack/{f=1} f' /tmp/m1_sample.txt | head -20   # opj_t1_* vs opj_dwt_* vs opj_mct_*
```

**Report from Phase 2:** OpenJPEG total decode ms on M1 at `-threads 1` and `-threads ALL`; and (if the
sample ran) the % in `opj_t1_*` so we can compute the real CPU **T1** time. With that we can answer:
does `max(CPU_T1, 37 ms GPU back-end)` fit the 41.6 ms budget (→ hybrid wins), or must T1 itself be sped up?

## Phase A-2 — overlapped hybrid harness (THE decision-gate measurement)

This is the end-to-end test: CPU decodes frame N+1 through Tier-1 (multithreaded, via the fork's
`opj_set_t1_output_callback`) while the GPU runs the back-end on frame N — overlapped — so per-frame
time = `max(CPU T1, GPU back-end)`. Needs OpenJPEG built (Phase 2) + a `.j2c` frame +
`proto/backend_corpus.bin` (for the integer-exact check).

```sh
swiftc -O proto/phase_a/hybrid_harness.swift \
  -import-objc-header proto/phase_a/bridge.h \
  -I src/lib/openjp2 -I build/src/lib/openjp2 \
  -L build/bin -lopenjp2 -Xlinker -rpath -Xlinker "$(pwd)/build/bin" \
  -framework Metal -framework Foundation -o /tmp/hybrid_harness

FRAME=path/to/2K-Flat_ProRes422_StereoVF_000000.j2c
/tmp/hybrid_harness "$FRAME" proto/phase_a/backend_opt.metal 120 proto/backend_corpus.bin
```

**Report:** the `OVERLAPPED hybrid` ms/frame + fps, the CPU/GPU calibration split, whether VALIDATION
is integer-exact, and whether it MEETS 24fps. This is the number that decides whether the hybrid
delivers realtime 2K on the M1. (M5 Pro reference: ~22 ms/frame, ~46 fps, integer-exact.)

### Optimized back-end (component-batched) — also measure this in isolation
The hybrid command above now uses the OPTIMIZED back-end (`proto/phase_a/backend_opt.metal`: 3
components batched into single H/V dispatches; integer-exact). To see the back-end speedup directly:
```sh
swiftc -O proto/phase_a/backend_opt_bench.swift -o /tmp/backend_opt_bench -framework Metal -framework Foundation
/tmp/backend_opt_bench proto/backend_corpus.bin proto/phase_a/backend_opt.metal
```
Report the `OPTIMIZED GPU back-end` ms/frame and compare to the earlier `/tmp/backend_bench` number
(M1 was ~38.5 ms; M5 Pro went 14.1 → 8.4 ms, ~1.7×). On the M1 the back-end was the binding stage, so
this drop should widen the hybrid's margin substantially.

### Threadgroup-memory back-end (FEASIBILITY §18 follow-up) — the M1's fix, measure it here
`proto/phase_a/backend_tg.metal` moves the DWT lifting scratch from device memory (`Wpool`) into
**threadgroup memory** — one threadgroup per line, DRAM touched only at gather/scatter. This targets
the bandwidth bound that made the M1 gain only 1.18× from component-batching, so it should help the
M1 the most. Same integer-exact bar (the bench validates vs the oracle).
```sh
swiftc -O proto/phase_a/backend_tg_bench.swift -o /tmp/backend_tg_bench -framework Metal -framework Foundation
/tmp/backend_tg_bench proto/backend_corpus.bin proto/phase_a/backend_tg.metal 256
# Optional: sweep the threads-per-line-threadgroup arg — report the best.
for T in 64 128 256 512; do /tmp/backend_tg_bench proto/backend_corpus.bin proto/phase_a/backend_tg.metal $T | tail -1; done
```
Report `THREADGROUP GPU back-end` ms/frame vs the `OPTIMIZED` number above, and the best `T`.
(M5 Pro reference: 8.08 → 1.30 ms at T=256, ~6×, integer-exact. M1 expectation: the 32.5 ms
optimized kernel should drop by a large factor — this is the number that says how much hybrid
headroom the M1 now has.) The same kernels ship in `swift-dcp-player`'s `HybridJ2KBackend`
(tgThreads=256; `KMQ_HYBRID_TG=0` reverts to the old kernels for in-app A/B).

## Regenerating corpora (only if the corpus files are missing)
The corpora are derived from one real DCI 2K frame via env-gated dumps in the (committed) instrumented
OpenJPEG. This requires building OpenJPEG (CMake + libpng/libtiff/lcms2) and a `.j2c` test frame —
heavier; prefer obtaining the prebuilt corpus files. If needed:
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=ON && cmake --build build -j
OPJ_T1_DUMP=proto/corpus.bin          DYLD_LIBRARY_PATH=build/bin build/bin/opj_decompress -i <frame>.j2c -o /tmp/x.tif -threads 1
OPJ_DWT_DUMP=proto/dwt_corpus.bin     DYLD_LIBRARY_PATH=build/bin build/bin/opj_decompress -i <frame>.j2c -o /tmp/x.tif -threads 1
OPJ_BACKEND_DUMP=proto/backend_corpus.bin DYLD_LIBRARY_PATH=build/bin build/bin/opj_decompress -i <frame>.j2c -o /tmp/x.tif -threads 1
```
