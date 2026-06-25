#include <metal_stdlib>
using namespace metal;

// LUTs (lut_ctxno_zc/sc/spb) are prepended from proto/t1_luts.metal by the harness.

#define T1_SIGMA_4   (1u << 4)
#define T1_SIGMA_NEIGHBOURS 0x1EFu
#define T1_CHI_1_I   19
#define T1_MU_0      (1u << 20)
#define T1_PI_0      (1u << 21)
#define T1_PI_1      (1u << 24)
#define T1_PI_2      (1u << 27)
#define T1_PI_3      (1u << 30)
#define T1_CHI_0_I   18
#define T1_CHI_5_I   31
#define T1_SIGMA_THIS T1_SIGMA_4
#define T1_PI_THIS    T1_PI_0
#define T1_MU_THIS    T1_MU_0
#define T1_CHI_THIS_I 19
#define T1_CTXNO_ZC  0
#define T1_CTXNO_SC  9
#define T1_CTXNO_MAG 14
#define T1_CTXNO_AGG 17
#define T1_CTXNO_UNI 18
#define MQC_NUMCTXS  19

struct mqstate_t { uint qeval; uint mps; uint nmps; uint nlps; };
constant mqstate_t S[94] = {
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

struct mqc_t {
    uint c, a, ct, eobsc;
    device const uchar* buf;
    uint bp, len;
    uint ctxs[MQC_NUMCTXS];
    uint curctx;
};

static void mq_bytein(thread mqc_t& m) {
    uint l_c = m.buf[m.bp + 1];
    if (m.buf[m.bp] == 0xff) {
        if (l_c > 0x8f) { m.c += 0xff00; m.ct = 8; m.eobsc++; }
        else { m.bp++; m.c += l_c << 9; m.ct = 7; }
    } else { m.bp++; m.c += l_c << 8; m.ct = 8; }
}
static void mq_init_dec(thread mqc_t& m, device const uchar* buf, uint len) {
    m.buf = buf; m.len = len; m.bp = 0; m.eobsc = 0; m.curctx = 0;
    m.c = (len == 0) ? (0xffu << 16) : ((uint)buf[0] << 16);
    mq_bytein(m);
    m.c <<= 7; m.ct -= 7; m.a = 0x8000;
}
static void mq_renormd(thread mqc_t& m) {
    do { if (m.ct == 0) mq_bytein(m); m.a <<= 1; m.c <<= 1; m.ct--; } while (m.a < 0x8000);
}
static uint mq_decode(thread mqc_t& m) {
    uint d; uint s = m.curctx; uint st = m.ctxs[s]; uint qeval = S[st].qeval;
    m.a -= qeval;
    if ((m.c >> 16) < qeval) {
        if (m.a < qeval) { m.a = qeval; d = S[st].mps;     m.ctxs[s] = S[st].nmps; }
        else             { m.a = qeval; d = 1 - S[st].mps; m.ctxs[s] = S[st].nlps; }
        mq_renormd(m);
    } else {
        m.c -= qeval << 16;
        if ((m.a & 0x8000) == 0) {
            if (m.a < qeval) { d = 1 - S[st].mps; m.ctxs[s] = S[st].nlps; }
            else             { d = S[st].mps;     m.ctxs[s] = S[st].nmps; }
            mq_renormd(m);
        } else { d = S[st].mps; }
    }
    return d;
}

static uint getctxno_zc(uint f, uint orient) {
    return lut_ctxno_zc[(orient << 9) + (f & T1_SIGMA_NEIGHBOURS)];
}
static uint getctxno_sc_or_spb_index(uint fX, uint pfX, uint nfX, uint ci) {
    uint lu = (fX >> (ci * 3u)) & ((1u<<1)|(1u<<3)|(1u<<5)|(1u<<7));
    lu |= (pfX >> (T1_CHI_THIS_I      + (ci * 3u))) & (1u << 0);
    lu |= (nfX >> (T1_CHI_THIS_I - 2u + (ci * 3u))) & (1u << 2);
    if (ci == 0u) lu |= (fX >> (T1_CHI_0_I - 4u)) & (1u << 4);
    else          lu |= (fX >> (T1_CHI_1_I - 4u + ((ci - 1u) * 3u))) & (1u << 4);
    lu |= (fX >> (22u - 6u + (ci * 3u))) & (1u << 6);
    return lu;
}
static uint getctxno_mag(uint f) {
    uint t = (f & T1_SIGMA_NEIGHBOURS) ? (T1_CTXNO_MAG + 1) : T1_CTXNO_MAG;
    return (f & T1_MU_0) ? (T1_CTXNO_MAG + 2) : t;
}
static void update_flags(thread uint& flags, device uint* flg, uint fp,
                         uint ci, uint s, uint stride) {
    flg[fp - 1] |= (1u << 5) << (3u * ci);
    flags |= ((s << T1_CHI_1_I) | T1_SIGMA_4) << (3u * ci);
    flg[fp + 1] |= (1u << 3) << (3u * ci);
    if (ci == 0u) {
        uint n = fp - stride;
        flg[n]     |= (s << T1_CHI_5_I) | (1u << 16);
        flg[n - 1] |= (1u << 17);
        flg[n + 1] |= (1u << 15);
    }
    if (ci == 3u) {
        uint so = fp + stride;
        flg[so]     |= (s << T1_CHI_0_I) | (1u << 1);
        flg[so - 1] |= (1u << 2);
        flg[so + 1] |= (1u << 0);
    }
}

static void dec_sigpass(thread mqc_t& m, device int* data, device uint* flg,
                        uint w, uint h, uint orient, int bpno) {
    int one = 1 << bpno, hf = one >> 1, oneplushalf = one | hf;
    uint stride = w + 2u, fp = stride + 1, dp = 0, i, j, k;
    for (k = 0; k + 4 <= h; k += 4, dp += 3 * w, fp += 2) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            uint flags = flg[fp];
            if (flags == 0) continue;
            for (uint ci = 0; ci < 4; ++ci) {
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3u))) == 0u &&
                    (flags & (T1_SIGMA_NEIGHBOURS << (ci*3u))) != 0u) {
                    m.curctx = getctxno_zc(flags >> (ci*3u), orient);
                    uint v = mq_decode(m);
                    if (v) {
                        uint lu = getctxno_sc_or_spb_index(flags, flg[fp-1], flg[fp+1], ci);
                        m.curctx = lut_ctxno_sc[lu];
                        v = mq_decode(m) ^ lut_spb[lu];
                        data[dp + ci*w] = v ? -oneplushalf : oneplushalf;
                        update_flags(flags, flg, fp, ci, v, stride);
                    }
                    flags |= T1_PI_THIS << (ci*3u);
                }
            }
            flg[fp] = flags;
        }
    }
    if (k < h) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            for (j = 0; j < h - k; ++j) {
                uint ci = j; uint flags = flg[fp];
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3u))) == 0u &&
                    (flags & (T1_SIGMA_NEIGHBOURS << (ci*3u))) != 0u) {
                    m.curctx = getctxno_zc(flags >> (ci*3u), orient);
                    uint v = mq_decode(m);
                    if (v) {
                        uint lu = getctxno_sc_or_spb_index(flags, flg[fp-1], flg[fp+1], ci);
                        m.curctx = lut_ctxno_sc[lu];
                        v = mq_decode(m) ^ lut_spb[lu];
                        data[dp + j*w] = v ? -oneplushalf : oneplushalf;
                        update_flags(flags, flg, fp, ci, v, stride);
                    }
                    flags |= T1_PI_THIS << (ci*3u);
                    flg[fp] = flags;
                }
            }
        }
    }
}

static void dec_refpass(thread mqc_t& m, device int* data, device uint* flg,
                        uint w, uint h, int bpno) {
    int one = 1 << bpno, poshalf = one >> 1;
    uint stride = w + 2u, fp = stride + 1, dp = 0, i, j, k;
    for (k = 0; k + 4 <= h; k += 4, dp += 3 * w, fp += 2) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            uint flags = flg[fp];
            if (flags == 0) continue;
            for (uint ci = 0; ci < 4; ++ci) {
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3u))) ==
                        (T1_SIGMA_THIS << (ci*3u))) {
                    m.curctx = getctxno_mag(flags >> (ci*3u));
                    uint v = mq_decode(m);
                    int dv = data[dp + ci*w];
                    data[dp + ci*w] = dv + ((v ^ (dv < 0 ? 1u : 0u)) ? poshalf : -poshalf);
                    flags |= T1_MU_THIS << (ci*3u);
                }
            }
            flg[fp] = flags;
        }
    }
    if (k < h) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            for (j = 0; j < h - k; ++j) {
                uint ci = j; uint flags = flg[fp];
                if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3u))) ==
                        (T1_SIGMA_THIS << (ci*3u))) {
                    m.curctx = getctxno_mag(flags >> (ci*3u));
                    uint v = mq_decode(m);
                    int dv = data[dp + j*w];
                    data[dp + j*w] = dv + ((v ^ (dv < 0 ? 1u : 0u)) ? poshalf : -poshalf);
                    flg[fp] |= T1_MU_THIS << (ci*3u);
                }
            }
        }
    }
}

static void cln_step(thread mqc_t& m, device int* data, device uint* flg, uint fp,
                     uint dp, uint w, uint orient, thread uint& flags, uint ci,
                     int oneplushalf, int check_flags, int partial) {
    if (!check_flags || !(flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci*3u)))) {
        if (!partial) {
            m.curctx = getctxno_zc(flags >> (ci*3u), orient);
            if (!mq_decode(m)) return;
        }
        uint lu = getctxno_sc_or_spb_index(flags, flg[fp-1], flg[fp+1], ci);
        m.curctx = lut_ctxno_sc[lu];
        uint v = mq_decode(m) ^ lut_spb[lu];
        data[dp + ci*w] = v ? -oneplushalf : oneplushalf;
        update_flags(flags, flg, fp, ci, v, w + 2u);
    }
}

static void dec_clnpass(thread mqc_t& m, device int* data, device uint* flg,
                        uint w, uint h, uint orient, int bpno) {
    int one = 1 << bpno, hf = one >> 1, oneplushalf = one | hf;
    uint stride = w + 2u, fp = stride + 1, dp = 0, i, j, k;
    for (k = 0; k + 4 <= h; k += 4, dp += 3 * w, fp += 2) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            uint flags = flg[fp];
            if (flags == 0) {
                int partial = 1;
                m.curctx = T1_CTXNO_AGG;
                if (!mq_decode(m)) continue;
                m.curctx = T1_CTXNO_UNI;
                uint runlen = mq_decode(m);
                uint vv = mq_decode(m);
                runlen = (runlen << 1) | vv;
                switch (runlen) {
                case 0: cln_step(m,data,flg,fp,dp,w,orient,flags,0,oneplushalf,0,1); partial=0; [[fallthrough]];
                case 1: cln_step(m,data,flg,fp,dp,w,orient,flags,1,oneplushalf,0,partial); partial=0; [[fallthrough]];
                case 2: cln_step(m,data,flg,fp,dp,w,orient,flags,2,oneplushalf,0,partial); partial=0; [[fallthrough]];
                case 3: cln_step(m,data,flg,fp,dp,w,orient,flags,3,oneplushalf,0,partial); break;
                }
            } else {
                cln_step(m,data,flg,fp,dp,w,orient,flags,0,oneplushalf,1,0);
                cln_step(m,data,flg,fp,dp,w,orient,flags,1,oneplushalf,1,0);
                cln_step(m,data,flg,fp,dp,w,orient,flags,2,oneplushalf,1,0);
                cln_step(m,data,flg,fp,dp,w,orient,flags,3,oneplushalf,1,0);
            }
            flg[fp] = flags & ~(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
        }
    }
    if (k < h) {
        for (i = 0; i < w; ++i, ++dp, ++fp) {
            for (j = 0; j < h - k; ++j) {
                uint flags = flg[fp];
                cln_step(m,data,flg,fp,dp,w,orient,flags,j,oneplushalf,1,0);
                flg[fp] = flags;
            }
            flg[fp] &= ~(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
        }
    }
}

static void init_flags(device uint* flg, uint w, uint h) {
    uint stride = w + 2u;
    uint flags_height = (h + 3u) / 4u;
    uint flagssize = ((h + 3u) / 4u + 2u) * stride;
    for (uint x = 0; x < flagssize; ++x) flg[x] = 0;
    for (uint x = 0; x < stride; ++x) flg[x] = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
    for (uint x = 0; x < stride; ++x)
        flg[(flags_height + 1) * stride + x] = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
    uint rem = h % 4;
    if (rem) {
        uint v = 0;
        if (rem == 1) v = T1_PI_1 | T1_PI_2 | T1_PI_3;
        else if (rem == 2) v = T1_PI_2 | T1_PI_3;
        else if (rem == 3) v = T1_PI_3;
        for (uint x = 0; x < stride; ++x) flg[flags_height * stride + x] = v;
    }
}

// desc layout per cblk (8 uints): dataOff, len, w, h, numbps, orient, passes, outOff
kernel void t1_decode(device const uchar* cdata   [[buffer(0)]],
                      device const uint*  desc    [[buffer(1)]],
                      device int*         out     [[buffer(2)]],
                      device uint*        flgpool  [[buffer(3)]],
                      constant uint&      ncblk    [[buffer(4)]],
                      constant uint&      flgstride [[buffer(5)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= ncblk) return;
    device const uint* d = desc + tid * 8u;
    uint dataOff = d[0], len = d[1], w = d[2], h = d[3];
    uint numbps = d[4], orient = d[5], passes = d[6], outOff = d[7];

    device int*  data = out + outOff;
    device uint* flg  = flgpool + tid * flgstride;
    device const uchar* buf = cdata + dataOff;

    for (uint q = 0; q < w*h; ++q) data[q] = 0;
    init_flags(flg, w, h);

    mqc_t m;
    int bpno_plus_one = (int)numbps;
    uint passtype = 2;
    for (uint q = 0; q < MQC_NUMCTXS; ++q) m.ctxs[q] = 0;
    mq_init_dec(m, buf, len);
    m.ctxs[T1_CTXNO_UNI] = 92;
    m.ctxs[T1_CTXNO_AGG] = 6;
    m.ctxs[T1_CTXNO_ZC]  = 8;

    for (uint passno = 0; passno < passes && bpno_plus_one >= 1; ++passno) {
        if (passtype == 0)      dec_sigpass(m, data, flg, w, h, orient, bpno_plus_one);
        else if (passtype == 1) dec_refpass(m, data, flg, w, h, bpno_plus_one);
        else                    dec_clnpass(m, data, flg, w, h, orient, bpno_plus_one);
        if (++passtype == 3) { passtype = 0; bpno_plus_one--; }
    }
}
