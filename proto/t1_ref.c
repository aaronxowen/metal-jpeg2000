/* Self-contained portable-C reference for JPEG2000 Tier-1 codeblock DECODE.
 *
 * Scope: the DCI / cblksty==0 case only — MQ coder, single segment, no
 * RAW/bypass, no RESTART, no vertical-stripe-causal, no segment-symbol.
 * This is a faithful port of OpenJPEG's opj_t1_decode_cblk + the three MQ
 * decode passes, written with plain arrays/indices (no pointers into a library)
 * so it translates almost mechanically to a Metal compute kernel.
 *
 * It reads proto/corpus.bin (produced by the OPJ_T1_DUMP hook in t1.c) and
 * validates every codeblock's decoded coefficients bit-exact against the oracle.
 *
 * Build:
 *   clang -O2 proto/t1_ref.c -o /tmp/t1_ref
 * Run:
 *   /tmp/t1_ref proto/corpus.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef uint8_t  OPJ_BYTE;
typedef int16_t  OPJ_INT16;
#define T1_NMSEDEC_BITS 7
#include "../src/lib/openjp2/t1_luts.h"   /* lut_ctxno_zc[2048], lut_ctxno_sc[256], lut_spb[256] */

/* ---- flag bit layout (from t1.h) ---- */
#define T1_SIGMA_4   (1U << 4)   /* THIS significant */
#define T1_SIGMA_NEIGHBOURS 0x1EFU
#define T1_CHI_1_I   19
#define T1_MU_0      (1U << 20)
#define T1_PI_0      (1U << 21)
#define T1_PI_1      (1U << 24)
#define T1_PI_2      (1U << 27)
#define T1_PI_3      (1U << 30)
#define T1_CHI_0_I   18
#define T1_CHI_5_I   31

#define T1_SIGMA_THIS T1_SIGMA_4
#define T1_PI_THIS    T1_PI_0
#define T1_MU_THIS    T1_MU_0
#define T1_CHI_THIS_I 19

/* context bases */
#define T1_CTXNO_ZC  0
#define T1_CTXNO_SC  9
#define T1_CTXNO_MAG 14
#define T1_CTXNO_AGG 17
#define T1_CTXNO_UNI 18
#define MQC_NUMCTXS  19

/* ---- MQ state table (transcribed from mqc.c mqc_states[47*2]) ---- */
typedef struct { uint32_t qeval; uint32_t mps; uint32_t nmps; uint32_t nlps; } mqstate_t;
static const mqstate_t S[94] = {
{0x5601,0,2,3},{0x5601,1,3,2},{0x3401,0,4,12},{0x3401,1,5,13},{0x1801,0,6,18},{0x1801,1,7,19},
{0x0ac1,0,8,24},{0x0ac1,1,9,25},{0x0521,0,10,58},{0x0521,1,11,59},{0x0221,0,76,66},{0x0221,1,77,67},
{0x5601,0,14,13},{0x5601,1,15,12},{0x5401,0,16,28},{0x5401,1,17,29},{0x4801,0,18,28},{0x4801,1,19,29},
{0x3801,0,20,28},{0x3801,1,21,29},{0x3001,0,22,34},{0x3001,1,23,35},{0x2401,0,24,36},{0x2401,1,25,37},
{0x1c01,0,26,40},{0x1c01,1,27,41},{0x1601,0,58,42},{0x1601,1,59,43},{0x5601,0,30,29},{0x5601,1,31,28},
{0x5401,0,32,28},{0x5401,1,33,29},{0x5101,0,34,30},{0x5101,1,35,31},{0x4801,0,36,32},{0x4801,1,37,33},
{0x3801,0,38,34},{0x3801,1,39,35},{0x3401,0,40,36},{0x3401,1,41,37},{0x3001,0,42,38},{0x3001,1,43,39},
{0x2801,0,44,38},{0x2801,1,45,39},{0x2401,0,46,40},{0x2401,1,47,41},{0x2201,0,48,42},{0x2201,1,49,43},
{0x1c01,0,50,44},{0x1c01,1,51,45},{0x1801,0,52,46},{0x1801,1,53,47},{0x1601,0,54,48},{0x1601,1,55,49},
{0x1401,0,56,50},{0x1401,1,57,51},{0x1201,0,58,52},{0x1201,1,59,53},{0x1101,0,60,54},{0x1101,1,61,55},
{0x0ac1,0,62,56},{0x0ac1,1,63,57},{0x09c1,0,64,58},{0x09c1,1,65,59},{0x08a1,0,66,60},{0x08a1,1,67,61},
{0x0521,0,68,62},{0x0521,1,69,63},{0x0441,0,70,64},{0x0441,1,71,65},{0x02a1,0,72,66},{0x02a1,1,73,67},
{0x0221,0,74,68},{0x0221,1,75,69},{0x0141,0,76,70},{0x0141,1,77,71},{0x0111,0,78,72},{0x0111,1,79,73},
{0x0085,0,80,74},{0x0085,1,81,75},{0x0049,0,82,76},{0x0049,1,83,77},{0x0025,0,84,78},{0x0025,1,85,79},
{0x0015,0,86,80},{0x0015,1,87,81},{0x0009,0,88,82},{0x0009,1,89,83},{0x0005,0,90,84},{0x0005,1,91,85},
{0x0001,0,90,86},{0x0001,1,91,87},{0x5601,0,92,92},{0x5601,1,93,93}
};

/* ---- MQ decoder state ---- */
typedef struct {
    uint32_t c, a, ct, eobsc;
    const uint8_t *buf;   /* len+2 bytes, buf[len]=buf[len+1]=0xFF */
    uint32_t bp, len;
    uint32_t ctxs[MQC_NUMCTXS];  /* current state index per context */
    uint32_t curctx;             /* active context slot */
} mqc_t;

static inline void mq_bytein(mqc_t *m) {
    uint32_t l_c = m->buf[m->bp + 1];
    if (m->buf[m->bp] == 0xff) {
        if (l_c > 0x8f) { m->c += 0xff00; m->ct = 8; m->eobsc++; }
        else { m->bp++; m->c += l_c << 9; m->ct = 7; }
    } else { m->bp++; m->c += l_c << 8; m->ct = 8; }
}

static void mq_init_dec(mqc_t *m, const uint8_t *buf, uint32_t len) {
    m->buf = buf; m->len = len; m->bp = 0; m->eobsc = 0; m->curctx = 0;
    m->c = (len == 0) ? (0xffU << 16) : ((uint32_t)buf[0] << 16);
    mq_bytein(m);
    m->c <<= 7; m->ct -= 7; m->a = 0x8000;
}

static inline void mq_renormd(mqc_t *m) {
    do {
        if (m->ct == 0) mq_bytein(m);
        m->a <<= 1; m->c <<= 1; m->ct--;
    } while (m->a < 0x8000);
}

static inline uint32_t mq_decode(mqc_t *m) {
    uint32_t d;
    uint32_t s = m->curctx;
    uint32_t st = m->ctxs[s];
    uint32_t qeval = S[st].qeval;
    m->a -= qeval;
    if ((m->c >> 16) < qeval) {            /* LPS exchange */
        if (m->a < qeval) { m->a = qeval; d = S[st].mps;     m->ctxs[s] = S[st].nmps; }
        else              { m->a = qeval; d = 1 - S[st].mps; m->ctxs[s] = S[st].nlps; }
        mq_renormd(m);
    } else {
        m->c -= qeval << 16;
        if ((m->a & 0x8000) == 0) {        /* MPS exchange */
            if (m->a < qeval) { d = 1 - S[st].mps; m->ctxs[s] = S[st].nlps; }
            else              { d = S[st].mps;     m->ctxs[s] = S[st].nmps; }
            mq_renormd(m);
        } else {
            d = S[st].mps;
        }
    }
    return d;
}

static void mq_resetstates(mqc_t *m) {
    for (int i = 0; i < MQC_NUMCTXS; i++) m->ctxs[i] = 0;
}

/* ---- context helpers ---- */
static inline uint32_t getctxno_zc(uint32_t f, uint32_t orient) {
    return lut_ctxno_zc[(orient << 9) + (f & T1_SIGMA_NEIGHBOURS)];
}
static inline uint32_t getctxno_sc_or_spb_index(uint32_t fX, uint32_t pfX, uint32_t nfX, uint32_t ci) {
    uint32_t lu = (fX >> (ci * 3U)) & ((1U<<1)|(1U<<3)|(1U<<5)|(1U<<7));
    lu |= (pfX >> (T1_CHI_THIS_I      + (ci * 3U))) & (1U << 0);
    lu |= (nfX >> (T1_CHI_THIS_I - 2U + (ci * 3U))) & (1U << 2);
    if (ci == 0U) lu |= (fX >> (T1_CHI_0_I - 4U)) & (1U << 4);
    else          lu |= (fX >> (T1_CHI_1_I - 4U + ((ci - 1U) * 3U))) & (1U << 4);
    lu |= (fX >> (22 - 6U + (ci * 3U))) & (1U << 6);   /* T1_CHI_2_I == 22 */
    return lu;
}
static inline uint32_t getctxno_mag(uint32_t f) {
    uint32_t t = (f & T1_SIGMA_NEIGHBOURS) ? (T1_CTXNO_MAG + 1) : T1_CTXNO_MAG;
    return (f & T1_MU_0) ? (T1_CTXNO_MAG + 2) : t;
}

/* update_flags: mirrors opj_t1_update_flags_macro (vsc always 0 here).
 * flg[] is the flags array, fp = index of current cell, *flags = local snapshot. */
static inline void update_flags(uint32_t *flags, uint32_t *flg, uint32_t fp,
                                uint32_t ci, uint32_t s, uint32_t stride) {
    flg[fp - 1] |= (1U << 5) << (3U * ci);                       /* east neighbour's SIGMA_5 ... wait: this cell's west */
    *flags |= ((s << T1_CHI_1_I) | T1_SIGMA_4) << (3U * ci);     /* mark self significant + sign */
    flg[fp + 1] |= (1U << 3) << (3U * ci);                       /* west neighbour SIGMA_3 */
    if (ci == 0U) {
        uint32_t n = fp - stride;
        flg[n]     |= (s << T1_CHI_5_I) | (1U << 16);
        flg[n - 1] |= (1U << 17);
        flg[n + 1] |= (1U << 15);
    }
    if (ci == 3U) {
        uint32_t so = fp + stride;
        flg[so]     |= (s << T1_CHI_0_I) | (1U << 1);
        flg[so - 1] |= (1U << 2);
        flg[so + 1] |= (1U << 0);
    }
}

/* ---- the three decode passes (generic, novsc, vsc==0) ---- */
/* data[] is w*h ints (row-major). flg[] is the flag buffer. */

static void dec_sigpass(mqc_t *m, int32_t *data, uint32_t *flg,
                        uint32_t w, uint32_t h, uint32_t orient, int bpno) {
    int one = 1 << bpno, half = one >> 1, oneplushalf = one | half;
    uint32_t stride = w + 2U;
    uint32_t fp = stride + 1;     /* &flags[stride+1] */
    uint32_t dp = 0;              /* data index */
    uint32_t i, j, k;
    for (k = 0; k + 4 <= h; k += 4, dp += 3 * w, fp += 2) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            uint32_t flags = flg[fp];
            if (flags == 0) continue;
            for (uint32_t ci = 0; ci < 4; ++ci) {
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3U))) == 0U &&
                    (flags & (T1_SIGMA_NEIGHBOURS << (ci*3U))) != 0U) {
                    m->curctx = getctxno_zc(flags >> (ci*3U), orient);
                    uint32_t v = mq_decode(m);
                    if (v) {
                        uint32_t lu = getctxno_sc_or_spb_index(flags, flg[fp-1], flg[fp+1], ci);
                        m->curctx = lut_ctxno_sc[lu];
                        v = mq_decode(m) ^ lut_spb[lu];
                        data[dp + ci*w] = v ? -oneplushalf : oneplushalf;
                        update_flags(&flags, flg, fp, ci, v, stride);
                    }
                    flags |= T1_PI_THIS << (ci*3U);
                }
            }
            flg[fp] = flags;
        }
    }
    /* tail rows */
    if (k < h) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            for (j = 0; j < h - k; ++j) {
                uint32_t ci = j;
                uint32_t flags = flg[fp];
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3U))) == 0U &&
                    (flags & (T1_SIGMA_NEIGHBOURS << (ci*3U))) != 0U) {
                    m->curctx = getctxno_zc(flags >> (ci*3U), orient);
                    uint32_t v = mq_decode(m);
                    if (v) {
                        uint32_t lu = getctxno_sc_or_spb_index(flags, flg[fp-1], flg[fp+1], ci);
                        m->curctx = lut_ctxno_sc[lu];
                        v = mq_decode(m) ^ lut_spb[lu];
                        data[dp + j*w] = v ? -oneplushalf : oneplushalf;
                        update_flags(&flags, flg, fp, ci, v, stride);
                    }
                    flags |= T1_PI_THIS << (ci*3U);
                    flg[fp] = flags;
                }
            }
        }
    }
}

static void dec_refpass(mqc_t *m, int32_t *data, uint32_t *flg,
                        uint32_t w, uint32_t h, int bpno) {
    int one = 1 << bpno, poshalf = one >> 1;
    uint32_t stride = w + 2U;
    uint32_t fp = stride + 1, dp = 0, i, j, k;
    for (k = 0; k + 4 <= h; k += 4, dp += 3 * w, fp += 2) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            uint32_t flags = flg[fp];
            if (flags == 0) continue;
            for (uint32_t ci = 0; ci < 4; ++ci) {
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3U))) ==
                        (T1_SIGMA_THIS << (ci*3U))) {
                    m->curctx = getctxno_mag(flags >> (ci*3U));
                    uint32_t v = mq_decode(m);
                    int32_t *d = &data[dp + ci*w];
                    *d += (v ^ (*d < 0)) ? poshalf : -poshalf;
                    flags |= T1_MU_THIS << (ci*3U);
                }
            }
            flg[fp] = flags;
        }
    }
    if (k < h) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            for (j = 0; j < h - k; ++j) {
                uint32_t ci = j, flags = flg[fp];
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3U))) ==
                        (T1_SIGMA_THIS << (ci*3U))) {
                    m->curctx = getctxno_mag(flags >> (ci*3U));
                    uint32_t v = mq_decode(m);
                    int32_t *d = &data[dp + j*w];
                    *d += (v ^ (*d < 0)) ? poshalf : -poshalf;
                    flg[fp] |= T1_MU_THIS << (ci*3U);
                }
            }
        }
    }
}

/* one cleanup-pass step (used by both the run path and the per-cell path) */
static inline void cln_step(mqc_t *m, int32_t *data, uint32_t *flg, uint32_t fp,
                            uint32_t dp, uint32_t w, uint32_t orient,
                            uint32_t *flags, uint32_t ci, int oneplushalf,
                            int check_flags, int partial) {
    if (!check_flags || !(*flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3U)))) {
        uint32_t v;
        if (!partial) {
            m->curctx = getctxno_zc(*flags >> (ci*3U), orient);
            if (!mq_decode(m)) return;
        }
        uint32_t lu = getctxno_sc_or_spb_index(*flags, flg[fp-1], flg[fp+1], ci);
        m->curctx = lut_ctxno_sc[lu];
        v = mq_decode(m) ^ lut_spb[lu];
        data[dp + ci*w] = v ? -oneplushalf : oneplushalf;
        update_flags(flags, flg, fp, ci, v, w + 2U);
    }
}

static void dec_clnpass(mqc_t *m, int32_t *data, uint32_t *flg,
                        uint32_t w, uint32_t h, uint32_t orient, int bpno) {
    int one = 1 << bpno, half = one >> 1, oneplushalf = one | half;
    uint32_t stride = w + 2U;
    uint32_t fp = stride + 1, dp = 0, i, j, k;
    for (k = 0; k + 4 <= h; k += 4, dp += 3 * w, fp += 2) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            uint32_t flags = flg[fp];
            if (flags == 0) {
                int partial = 1;
                m->curctx = T1_CTXNO_AGG;
                if (!mq_decode(m)) continue;
                m->curctx = T1_CTXNO_UNI;
                uint32_t runlen = mq_decode(m);
                uint32_t vv = mq_decode(m);
                runlen = (runlen << 1) | vv;
                switch (runlen) {
                case 0: cln_step(m,data,flg,fp,dp,w,orient,&flags,0,oneplushalf,0,1); partial=0; /*FALL*/
                case 1: cln_step(m,data,flg,fp,dp,w,orient,&flags,1,oneplushalf,0,partial); partial=0; /*FALL*/
                case 2: cln_step(m,data,flg,fp,dp,w,orient,&flags,2,oneplushalf,0,partial); partial=0; /*FALL*/
                case 3: cln_step(m,data,flg,fp,dp,w,orient,&flags,3,oneplushalf,0,partial); break;
                }
            } else {
                cln_step(m,data,flg,fp,dp,w,orient,&flags,0,oneplushalf,1,0);
                cln_step(m,data,flg,fp,dp,w,orient,&flags,1,oneplushalf,1,0);
                cln_step(m,data,flg,fp,dp,w,orient,&flags,2,oneplushalf,1,0);
                cln_step(m,data,flg,fp,dp,w,orient,&flags,3,oneplushalf,1,0);
            }
            flg[fp] = flags & ~(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
        }
    }
    if (k < h) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            for (j = 0; j < h - k; ++j) {
                uint32_t flags = flg[fp];
                cln_step(m,data,flg,fp,dp,w,orient,&flags,j,oneplushalf,1,0);
                flg[fp] = flags;
            }
            flg[fp] &= ~(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
        }
    }
}

/* allocate + init flags exactly like opj_t1_allocate_buffers */
static void init_flags(uint32_t *flg, uint32_t w, uint32_t h) {
    uint32_t stride = w + 2U;
    uint32_t flags_height = (h + 3U) / 4U;
    uint32_t flagssize = ((h + 3U) / 4U + 2U) * stride;
    memset(flg, 0, flagssize * sizeof(uint32_t));
    for (uint32_t x = 0; x < stride; ++x)
        flg[x] = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
    for (uint32_t x = 0; x < stride; ++x)
        flg[(flags_height + 1) * stride + x] = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
    if (h % 4) {
        uint32_t v = 0;
        if (h % 4 == 1) v = T1_PI_1 | T1_PI_2 | T1_PI_3;
        else if (h % 4 == 2) v = T1_PI_2 | T1_PI_3;
        else if (h % 4 == 3) v = T1_PI_3;
        for (uint32_t x = 0; x < stride; ++x) flg[flags_height * stride + x] = v;
    }
}

/* decode one codeblock (cblksty==0, single segment). buf has len+2 bytes. */
static void decode_cblk(const uint8_t *seg, uint32_t len, uint32_t w, uint32_t h,
                        uint32_t numbps, uint32_t orient, uint32_t real_num_passes,
                        int32_t *data, uint32_t *flg) {
    uint8_t *buf = (uint8_t*)malloc(len + 2);
    memcpy(buf, seg, len); buf[len] = 0xff; buf[len + 1] = 0xff;
    memset(data, 0, (size_t)w * h * sizeof(int32_t));
    init_flags(flg, w, h);

    mqc_t m;
    int bpno_plus_one = (int)numbps;
    uint32_t passtype = 2;
    mq_resetstates(&m);
    /* curctx slots set after init; init_dec sets curctx=0, but each pass sets it */
    mq_init_dec(&m, buf, len);
    m.ctxs[T1_CTXNO_UNI] = 92;  /* setstate(UNI,0,46) */
    m.ctxs[T1_CTXNO_AGG] = 6;   /* setstate(AGG,0,3)  */
    m.ctxs[T1_CTXNO_ZC]  = 8;   /* setstate(ZC,0,4)   */

    for (uint32_t passno = 0; passno < real_num_passes && bpno_plus_one >= 1; ++passno) {
        switch (passtype) {
        case 0: dec_sigpass(&m, data, flg, w, h, orient, bpno_plus_one); break;
        case 1: dec_refpass(&m, data, flg, w, h, bpno_plus_one); break;
        case 2: dec_clnpass(&m, data, flg, w, h, orient, bpno_plus_one); break;
        }
        if (++passtype == 3) { passtype = 0; bpno_plus_one--; }
    }
    free(buf);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "proto/corpus.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open corpus"); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *all = malloc(fsz); fread(all, 1, fsz, f); fclose(f);

    int32_t *data = malloc(64 * 64 * sizeof(int32_t));
    uint32_t *flg = malloc((64 / 4 + 2) * (64 + 2) * sizeof(uint32_t));
    int32_t *out = malloc(64 * 64 * sizeof(int32_t));  /* decoded */

    long off = 0; int n = 0, fail = 0;
    long total_coeffs = 0;
    struct timespec t0, t1;
    /* read all records into memory first so timing excludes I/O */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (off < fsz) {
        uint32_t *hdr = (uint32_t*)(all + off); off += 28;
        uint32_t orient = hdr[0], roishift = hdr[1], cblksty = hdr[2];
        uint32_t numbps = hdr[3], w = hdr[4], h = hdr[5], nseg = hdr[6];
        uint32_t passes = 0;
        for (uint32_t s = 0; s < nseg; ++s) {
            uint32_t *seg = (uint32_t*)(all + off); off += 8;
            passes += seg[1];
        }
        uint32_t tot = *(uint32_t*)(all + off); off += 4;
        const uint8_t *segdata = all + off; off += tot;
        const int32_t *oracle = (int32_t*)(all + off); off += 4 * w * h;
        (void)roishift; (void)cblksty;

        decode_cblk(segdata, tot, w, h, numbps, orient, passes, out, flg);
        if (memcmp(out, oracle, (size_t)w * h * sizeof(int32_t)) != 0) {
            if (fail < 5) {
                int diffs = 0, first = -1;
                for (uint32_t k = 0; k < w*h; ++k) if (out[k] != oracle[k]) { diffs++; if(first<0) first=(int)k; }
                printf("MISMATCH cblk #%d w=%u h=%u numbps=%u orient=%u passes=%u: %d/%u coeffs differ, first @%d (got %d want %d)\n",
                       n, w, h, numbps, orient, passes, diffs, w*h, first, out[first], oracle[first]);
            }
            fail++;
        }
        total_coeffs += w * h;
        n++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    (void)data;
    printf("\ncodeblocks: %d   PASS: %d   FAIL: %d\n", n, n - fail, fail);
    printf("CPU reference T1 decode: %.2f ms for %d cblks (%.1f us/cblk, %ld coeffs)\n",
           ms, n, ms*1000.0/n, total_coeffs);
    return fail ? 2 : 0;
}
