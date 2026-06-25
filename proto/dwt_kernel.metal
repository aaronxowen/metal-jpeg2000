#include <metal_stdlib>
using namespace metal;

constant float K = 1.230174105f;
constant float two_invK = 1.625732422f;
constant float alpha = -1.586134342f;
constant float beta  = -0.052980118f;
constant float gamma_ = 0.882911075f;
constant float delta = 0.443506852f;

static int imin(int a, int b){ return a < b ? a : b; }

static void step1(device float* base, int start, int end, float c) {
    for (int i = start; i < end; ++i) base[i*2] *= c;
}
static void step2(device float* lbase, device float* wbase, int start, int end, int m, float c) {
    device float* fl = lbase; device float* fw = wbase;
    int imax = imin(end, m);
    if (start > 0) { fw += 2*start; fl = fw - 2; }
    for (int i = start; i < imax; ++i) {
        fw[-1] = fw[-1] + (fl[0] + fw[0]) * c;
        fl = fw; fw += 2;
    }
    if (m < end) { c += c; fw[-1] = fw[-1] + fl[0] * c; }
}
static void idwt97_1d(device float* W, int sn, int dn, int cas) {
    int a, b;
    if (cas == 0) { if (!((dn > 0) || (sn > 1))) return; a = 0; b = 1; }
    else          { if (!((sn > 0) || (dn > 1))) return; a = 1; b = 0; }
    step1(W + a, 0, sn, K);
    step1(W + b, 0, dn, two_invK);
    step2(W + b, W + a + 1, 0, sn, imin(sn, dn - a), -delta);
    step2(W + a, W + b + 1, 0, dn, imin(dn, sn - b), -gamma_);
    step2(W + b, W + a + 1, 0, sn, imin(sn, dn - a), -beta);
    step2(W + a, W + b + 1, 0, dn, imin(dn, sn - b), -alpha);
}

// boxes: 4 ints per resolution (x0,y0,x1,y1). lvl in [1, numres).
// Horizontal pass: one thread per row j in [0, rh).
kernel void idwt97_h(device float* data       [[buffer(0)]],
                     device const int* boxes  [[buffer(1)]],
                     device float* Wpool      [[buffer(2)]],
                     constant uint& w         [[buffer(3)]],
                     constant uint& lvl       [[buffer(4)]],
                     constant uint& wstride   [[buffer(5)]],
                     uint tid [[thread_position_in_grid]]) {
    int sn_h = boxes[(lvl-1)*4+2] - boxes[(lvl-1)*4+0];
    int x0 = boxes[lvl*4+0];
    int rw = boxes[lvl*4+2] - x0;
    int rh = boxes[lvl*4+3] - boxes[lvl*4+1];
    if ((int)tid >= rh) return;
    int dn_h = rw - sn_h, cas_h = x0 & 1;
    device float* row = data + (size_t)tid * w;
    device float* W = Wpool + (size_t)tid * wstride;
    for (int i = 0; i < sn_h; ++i) W[2*i + cas_h]     = row[i];
    for (int i = 0; i < dn_h; ++i) W[2*i + (1-cas_h)] = row[sn_h + i];
    idwt97_1d(W, sn_h, dn_h, cas_h);
    for (int k = 0; k < rw; ++k) row[k] = W[k];
}

// Vertical pass: one thread per column i in [0, rw).
kernel void idwt97_v(device float* data       [[buffer(0)]],
                     device const int* boxes  [[buffer(1)]],
                     device float* Wpool      [[buffer(2)]],
                     constant uint& w         [[buffer(3)]],
                     constant uint& lvl       [[buffer(4)]],
                     constant uint& wstride   [[buffer(5)]],
                     uint tid [[thread_position_in_grid]]) {
    int sn_v = boxes[(lvl-1)*4+3] - boxes[(lvl-1)*4+1];
    int y0 = boxes[lvl*4+1];
    int rw = boxes[lvl*4+2] - boxes[lvl*4+0];
    int rh = boxes[lvl*4+3] - y0;
    if ((int)tid >= rw) return;
    int dn_v = rh - sn_v, cas_v = y0 & 1;
    device float* W = Wpool + (size_t)tid * wstride;
    for (int k = 0; k < sn_v; ++k) W[2*k + cas_v]     = data[(size_t)k * w + tid];
    for (int k = 0; k < dn_v; ++k) W[2*k + (1-cas_v)] = data[(size_t)(sn_v + k) * w + tid];
    idwt97_1d(W, sn_v, dn_v, cas_v);
    for (int k = 0; k < rh; ++k) data[(size_t)k * w + tid] = W[k];
}
