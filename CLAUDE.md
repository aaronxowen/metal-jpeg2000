# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project context (metal-jpeg2000)

**What this repo is:** a fork of [`uclouvain/openjpeg`](https://github.com/uclouvain/openjpeg) (`upstream`) repurposed as a **GPU/Metal JPEG 2000 decode feasibility project** — making realtime DCP playback possible on a low-floor M1 Mac mini. It is *not* greenfield; most of the tree is upstream OpenJPEG. Read these first:
- `FEASIBILITY.md` — the study (exec summary up top; §17–18 are the most current results).
- `swift-player-integration-scope.md`, `phase-c-integration-scope.md` — the integration plan.
- `proto/` — the prototype (per-stage reference C + Metal kernels + benches); see `proto/README.md`.

**Architecture decided:** hybrid decode — CPU runs Tier-2 + Tier-1 (the serial MQ coder, faster on CPU), the GPU runs the back-end (inverse DWT + ICT + level-shift, and color), overlapped across frames via unified memory. GPU Tier-1 was tried and abandoned (slower than CPU).

**Git hygiene:** never `git add -A`. `build/`, `proto/*corpus.bin`, and `tests/test-content/` (~424 MB) are large and gitignored — stage explicit files only.

**Build gotchas (correctness-load-bearing):**
- Metal benchmarks must compile with **fast-math OFF**; C reference decoders with **`-ffp-contract=off`** — both to match OpenJPEG's non-fused NEON path and stay bit/float-exact.
- CMake via the official Kitware binary, not Homebrew (the `cirruslabs` tap breaks `brew install`).

**Validation discipline:** every GPU stage is validated **bit-exact (T1) / integer-exact (back-end)** against an OpenJPEG-derived oracle (methodology: instrument opj to dump a corpus + oracle → portable-C reference → Metal kernel → exact compare). Preserve that bar — do not loosen to "looks right".

**Hooks / APIs added by the fork:** env-gated corpus dumps `OPJ_T1_DUMP` / `OPJ_DWT_DUMP` / `OPJ_BACKEND_DUMP` (decode with `-threads 1`); the decode-to-Tier-1 callback `opj_set_t1_output_callback` (`openjpeg.h` / `tcd.c`).

**Performance target is the M1 Mac mini**, not the dev machine — measure there. The GPU benches are self-contained (need only a corpus file + `swiftc`); see `proto/M1_RUNBOOK.md`. Cross-machine results land in `proto/M1_RESULTS.md` via the branch on origin.


