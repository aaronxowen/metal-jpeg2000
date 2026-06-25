/* Portable-C reference for the inverse 9/7 (irreversible) DWT.
 * Faithful scalar port of OpenJPEG's opj_dwt_decode_tile_97 / opj_v8dwt_decode
 * (NB_ELTS = 1). Reads proto/dwt_corpus.bin (OPJ_DWT_DUMP) and validates the
 * reconstructed float buffer against the oracle.
 *
 * Build: clang -O2 -ffp-contract=off proto/dwt_ref.c -o /tmp/dwt_ref
 *   (-ffp-contract=off matches OpenJPEG's non-fused NEON vmlaq_f32 path)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static const float K = 1.230174105f;
static const float two_invK = 1.625732422f;
static const float alpha = -1.586134342f;
static const float beta  = -0.052980118f;
static const float gamma_ = 0.882911075f;
static const float delta = 0.443506852f;

static inline int imin(int a, int b){ return a < b ? a : b; }

/* step1: base[i*2] *= c for i in [start,end) */
static void step1(float* base, int start, int end, float c) {
    for (int i = start; i < end; ++i) base[i*2] *= c;
}
/* step2: mirrors opj_v8dwt_decode_step2 with NB_ELTS=1 (stride 2, fw[-1]) */
static void step2(float* lbase, float* wbase, int start, int end, int m, float c) {
    float* fl = lbase; float* fw = wbase;
    int imax = imin(end, m);
    if (start > 0) { fw += 2*start; fl = fw - 2; }
    for (int i = start; i < imax; ++i) {
        fw[-1] = fw[-1] + (fl[0] + fw[0]) * c;
        fl = fw; fw += 2;
    }
    if (m < end) { c += c; fw[-1] = fw[-1] + fl[0] * c; }
}
/* 1-D inverse 9/7 on interleaved buffer W (sn low + dn high samples) */
static void idwt97_1d(float* W, int sn, int dn, int cas) {
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

/* full 2-D multi-level inverse 9/7 on buf (w*h, top-left rw*rh per level) */
static void idwt97_2d(float* buf, uint32_t w, const int32_t* boxes, uint32_t numres, float* W) {
    int rw = boxes[0*4+2] - boxes[0*4+0];
    int rh = boxes[0*4+3] - boxes[0*4+1];
    for (uint32_t lvl = 1; lvl < numres; ++lvl) {
        int sn_h = rw, sn_v = rh;
        int x0 = boxes[lvl*4+0], y0 = boxes[lvl*4+1];
        rw = boxes[lvl*4+2] - x0;
        rh = boxes[lvl*4+3] - y0;
        int dn_h = rw - sn_h, cas_h = x0 & 1;
        int dn_v = rh - sn_v, cas_v = y0 & 1;
        /* horizontal: each of rh rows */
        for (int j = 0; j < rh; ++j) {
            float* row = buf + (size_t)j * w;
            for (int i = 0; i < sn_h; ++i) W[2*i + cas_h]     = row[i];
            for (int i = 0; i < dn_h; ++i) W[2*i + (1-cas_h)] = row[sn_h + i];
            idwt97_1d(W, sn_h, dn_h, cas_h);
            for (int k = 0; k < rw; ++k) row[k] = W[k];
        }
        /* vertical: each of rw columns */
        for (int i = 0; i < rw; ++i) {
            for (int k = 0; k < sn_v; ++k) W[2*k + cas_v]     = buf[(size_t)k * w + i];
            for (int k = 0; k < dn_v; ++k) W[2*k + (1-cas_v)] = buf[(size_t)(sn_v + k) * w + i];
            idwt97_1d(W, sn_v, dn_v, cas_v);
            for (int k = 0; k < rh; ++k) buf[(size_t)k * w + i] = W[k];
        }
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "proto/dwt_corpus.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* all = malloc(fsz); fread(all, 1, fsz, f); fclose(f);

    long off = 0; int rec = 0;
    double maxabs = 0.0, maxrel = 0.0; long ndiff = 0, ntot = 0;
    double ms_total = 0.0;
    while (off < fsz) {
        uint32_t magic; memcpy(&magic, all+off, 4); off += 4;
        if (magic != 0x31545744) { printf("bad magic @%ld\n", off); break; }
        uint32_t numres; memcpy(&numres, all+off, 4); off += 4;
        int32_t* boxes = (int32_t*)(all+off); off += 16 * numres;
        uint32_t w, h; memcpy(&w, all+off, 4); memcpy(&h, all+off+4, 4); off += 8;
        size_t n = (size_t)w * h;
        float* input = (float*)(all+off); off += 4*n;
        float* oracle = (float*)(all+off); off += 4*n;

        float* buf = malloc(n * sizeof(float));
        memcpy(buf, input, n * sizeof(float));
        float* W = malloc((size_t)(w > h ? w : h) * 2 * sizeof(float) + 64);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        idwt97_2d(buf, w, boxes, numres, W);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms_total += (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6;

        for (size_t i = 0; i < n; ++i) {
            float a = buf[i], b = oracle[i];
            double d = fabs((double)a - (double)b);
            if (d != 0.0) {
                ndiff++;
                if (d > maxabs) maxabs = d;
                double rel = d / (fabs((double)b) + 1e-6);
                if (rel > maxrel) maxrel = rel;
            }
        }
        ntot += n;
        free(buf); free(W);
        rec++;
    }
    printf("records: %d   coeffs: %ld   exact: %ld   differ: %ld\n", rec, ntot, ntot-ndiff, ndiff);
    printf("max abs diff: %.6g   max rel diff: %.6g\n", maxabs, maxrel);
    printf("CPU reference iDWT: %.2f ms total (%.2f ms/component)\n", ms_total, ms_total/rec);
    int ok = (maxabs == 0.0);
    printf("%s\n", ok ? "FLOAT-EXACT vs oracle" : (maxabs < 1e-2 ? "within tolerance (rounds to same int image)" : "OUT OF TOLERANCE"));
    return (maxabs < 1e-2) ? 0 : 2;
}
