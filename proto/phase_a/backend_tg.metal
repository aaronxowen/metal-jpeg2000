#include <metal_stdlib>
using namespace metal;

// Threadgroup-memory back-end (FEASIBILITY §18 follow-up): one THREADGROUP per line, the
// lifting scratch W in threadgroup memory. The six lifting sub-passes run in on-chip memory;
// DRAM is touched only at the initial gather and final scatter — targeting the bandwidth
// bound that limited backend_opt.metal's win on the M1 (1.18× vs the M5's 1.68×).
//
// Each sub-pass writes one parity of W and reads the other, so a pass distributes over the
// threadgroup (barrier between passes) with per-element arithmetic IDENTICAL to the serial
// idwt97_1d — still integer-exact. finalize_b is unchanged from backend_opt.metal.

constant float K = 1.230174105f;
constant float two_invK = 1.625732422f;
constant float alpha = -1.586134342f;
constant float beta  = -0.052980118f;
constant float gamma_ = 0.882911075f;
constant float delta = 0.443506852f;

static int imin(int a, int b){ return a < b ? a : b; }

static void step1_tg(threadgroup float* base, int end, float c, uint lid, uint tsz) {
    for (int i = (int)lid; i < end; i += (int)tsz) base[i*2] *= c;
}
static void step2_tg(threadgroup float* lbase, threadgroup float* wbase, int end, int m,
                     float c, uint lid, uint tsz) {
    int imax = imin(end, m);
    for (int i = (int)lid; i < imax; i += (int)tsz) {
        threadgroup float* fl = (i == 0) ? lbase : (wbase + 2*i - 2);
        wbase[2*i - 1] = wbase[2*i - 1] + (fl[0] + wbase[2*i]) * c;
    }
    // Tail (m < end): same write parity, distinct element — safe in the same pass.
    if (m < end && lid == 0) {
        threadgroup float* fl = (imax == 0) ? lbase : (wbase + 2*imax - 2);
        wbase[2*imax - 1] = wbase[2*imax - 1] + fl[0] * (c + c);
    }
}
static void idwt97_1d_tg(threadgroup float* W, int sn, int dn, int cas, uint lid, uint tsz) {
    int a, b;
    if (cas == 0) { if (!((dn > 0) || (sn > 1))) return; a = 0; b = 1; }
    else          { if (!((sn > 0) || (dn > 1))) return; a = 1; b = 0; }
    step1_tg(W + a, sn, K, lid, tsz);          // disjoint parities — no barrier between
    step1_tg(W + b, dn, two_invK, lid, tsz);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step2_tg(W + b, W + a + 1, sn, imin(sn, dn - a), -delta, lid, tsz);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step2_tg(W + a, W + b + 1, dn, imin(dn, sn - b), -gamma_, lid, tsz);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step2_tg(W + b, W + a + 1, sn, imin(sn, dn - a), -beta, lid, tsz);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step2_tg(W + a, W + b + 1, dn, imin(dn, sn - b), -alpha, lid, tsz);
}

// Horizontal pass: one threadgroup per (comp,row); W sized to the level's rw floats.
kernel void idwt97_h_tg(device float* data        [[buffer(0)]],
                        device const int* boxes   [[buffer(1)]],
                        constant uint& w          [[buffer(2)]],
                        constant uint& lvl        [[buffer(3)]],
                        constant uint& compStride [[buffer(4)]],
                        constant uint& rh_        [[buffer(5)]],
                        threadgroup float* W [[threadgroup(0)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tsz [[threads_per_threadgroup]]) {
    uint comp = gid / rh_, row = gid % rh_;
    int sn_h = boxes[(lvl-1)*4+2] - boxes[(lvl-1)*4+0];
    int x0 = boxes[lvl*4+0];
    int rw = boxes[lvl*4+2] - x0;
    int dn_h = rw - sn_h, cas_h = x0 & 1;
    device float* rowp = data + (size_t)comp * compStride + (size_t)row * w;
    for (int i = (int)lid; i < sn_h; i += (int)tsz) W[2*i + cas_h]     = rowp[i];
    for (int i = (int)lid; i < dn_h; i += (int)tsz) W[2*i + (1-cas_h)] = rowp[sn_h + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    idwt97_1d_tg(W, sn_h, dn_h, cas_h, lid, tsz);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = (int)lid; k < rw; k += (int)tsz) rowp[k] = W[k];
}

// Vertical pass: one threadgroup per (comp,col); W sized to the level's rh floats.
kernel void idwt97_v_tg(device float* data        [[buffer(0)]],
                        device const int* boxes   [[buffer(1)]],
                        constant uint& w          [[buffer(2)]],
                        constant uint& lvl        [[buffer(3)]],
                        constant uint& compStride [[buffer(4)]],
                        constant uint& rw_        [[buffer(5)]],
                        threadgroup float* W [[threadgroup(0)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tsz [[threads_per_threadgroup]]) {
    uint comp = gid / rw_, col = gid % rw_;
    int sn_v = boxes[(lvl-1)*4+3] - boxes[(lvl-1)*4+1];
    int y0 = boxes[lvl*4+1];
    int rh = boxes[lvl*4+3] - y0;
    int dn_v = rh - sn_v, cas_v = y0 & 1;
    device float* base = data + (size_t)comp * compStride;
    for (int k = (int)lid; k < sn_v; k += (int)tsz) W[2*k + cas_v]     = base[(size_t)k * w + col];
    for (int k = (int)lid; k < dn_v; k += (int)tsz) W[2*k + (1-cas_v)] = base[(size_t)(sn_v + k) * w + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    idwt97_1d_tg(W, sn_v, dn_v, cas_v, lid, tsz);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = (int)lid; k < rh; k += (int)tsz) base[(size_t)k * w + col] = W[k];
}

struct FinalizeParams { uint n; int mct; int dc[4]; int lo[4]; int hi[4]; };

// Inverse ICT + DC level-shift + clamp. 1 thread/pixel; in/out are concatenated (compStride).
kernel void finalize_b(device const float* data   [[buffer(0)]],
                       device int* out            [[buffer(1)]],
                       constant FinalizeParams& p  [[buffer(2)]],
                       constant uint& compStride    [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= p.n) return;
    float y = data[tid], u = data[compStride + tid], v = data[2*compStride + tid];
    float r, g, b;
    if (p.mct) { r = y + v*1.402f; g = y - u*0.34413f - v*0.71414f; b = y + u*1.772f; }
    else { r = y; g = u; b = v; }
    out[tid]               = clamp((int)rint(r) + p.dc[0], p.lo[0], p.hi[0]);
    out[compStride + tid]  = clamp((int)rint(g) + p.dc[1], p.lo[1], p.hi[1]);
    out[2*compStride + tid]= clamp((int)rint(b) + p.dc[2], p.lo[2], p.hi[2]);
}
