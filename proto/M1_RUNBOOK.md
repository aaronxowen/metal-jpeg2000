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
