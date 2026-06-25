/* Phase A, step 1 — validate the decode-to-Tier-1 entry point.
 *
 * Registers opj_set_t1_output_callback (skip_backend=TRUE), decodes a real DCI
 * 2K .j2c, and checks that the post-T1 buffers + geometry handed to the callback
 * are byte-identical to backend_corpus.bin's INPUT section (which OPJ_BACKEND_DUMP
 * captured at the same split point, and against which the GPU back-end is already
 * integer-exact). Match => the decode-to-T1 API feeds the proven GPU back-end.
 *
 * Build:
 *   clang -O2 -I src/lib/openjp2 -I build/src/lib/openjp2 proto/phase_a/t1extract_test.c \
 *     -o /tmp/t1extract_test -Lbuild/bin -lopenjp2 -Wl,-rpath,$(pwd)/build/bin
 * Run:
 *   /tmp/t1extract_test <frame>.j2c proto/backend_corpus.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "openjpeg.h"

typedef struct {
    uint32_t nc, w, h, numres, mct;
    const int32_t* boxes;     /* numres*4 */
    const int32_t* prec; const int32_t* sgnd; const int32_t* dcs;
    const float*   in[8];     /* expected post-T1 buffers */
    int fired, mismatches;
} Expected;

static void quiet(const char* m, void* d){ (void)m;(void)d; }

static void on_t1(const opj_t1_output_t* o, void* user) {
    Expected* e = (Expected*)user;
    e->fired++;
    int bad = 0;
    if (o->numcomps != e->nc || o->w != e->w || o->h != e->h || o->numres != e->numres) {
        printf("  geometry mismatch: nc %u/%u w %u/%u h %u/%u numres %u/%u\n",
               o->numcomps, e->nc, o->w, e->w, o->h, e->h, o->numres, e->numres); bad++;
    }
    if (o->mct != (int32_t)e->mct) { printf("  mct mismatch: %d/%u\n", o->mct, e->mct); bad++; }
    for (uint32_t r = 0; r < o->numres && r < e->numres; ++r)
        for (int k = 0; k < 4; ++k)
            if (o->boxes[r*4+k] != e->boxes[r*4+k]) { printf("  box[%u][%d] mismatch\n", r, k); bad++; }
    size_t n = (size_t)e->w * e->h;
    for (uint32_t c = 0; c < o->numcomps && c < e->nc; ++c) {
        if (o->prec[c] != e->prec[c] || o->sgnd[c] != e->sgnd[c] || o->dc_shift[c] != e->dcs[c]) {
            printf("  comp %u meta mismatch\n", c); bad++;
        }
        /* compare post-T1 coefficients bit-for-bit */
        if (memcmp(o->comp_data[c], e->in[c], n * sizeof(int32_t)) != 0) {
            size_t diff = 0; for (size_t i = 0; i < n; ++i) if (o->comp_data[c][i] != ((const int32_t*)e->in[c])[i]) diff++;
            printf("  comp %u data mismatch: %zu/%zu coeffs differ\n", c, diff, n); bad++;
        }
    }
    e->mismatches += bad;
}

int main(int argc, char** argv) {
    const char* j2c = argc > 1 ? argv[1] : NULL;
    const char* corpus = argc > 2 ? argv[2] : "proto/backend_corpus.bin";
    if (!j2c) { fprintf(stderr, "usage: %s <frame>.j2c [backend_corpus.bin]\n", argv[0]); return 1; }

    /* Load expected post-T1 data from the back-end corpus input section. */
    FILE* cf = fopen(corpus, "rb");
    if (!cf) { perror("corpus"); return 1; }
    fseek(cf, 0, SEEK_END); long csz = ftell(cf); fseek(cf, 0, SEEK_SET);
    uint8_t* cb = malloc(csz); fread(cb, 1, csz, cf); fclose(cf);
    long off = 0; Expected e; memset(&e, 0, sizeof e);
    uint32_t magic; memcpy(&magic, cb+off, 4); off += 4;
    if (magic != 0x444E4B42) { fprintf(stderr, "bad corpus magic\n"); return 1; }
    memcpy(&e.nc, cb+off, 4); off += 4;
    memcpy(&e.w, cb+off, 4); memcpy(&e.h, cb+off+4, 4); off += 8;
    memcpy(&e.numres, cb+off, 4); off += 4;
    e.boxes = (int32_t*)(cb+off); off += 16 * e.numres;
    e.prec = (int32_t*)(cb+off);
    /* prec/sgnd/dcs are interleaved 3 ints per comp in the corpus; de-interleave */
    static int32_t prec[16], sgnd[16], dcs[16];
    for (uint32_t c = 0; c < e.nc; ++c) { prec[c]=*(int32_t*)(cb+off); sgnd[c]=*(int32_t*)(cb+off+4); dcs[c]=*(int32_t*)(cb+off+8); off += 12; }
    e.prec = prec; e.sgnd = sgnd; e.dcs = dcs;
    memcpy(&e.mct, cb+off, 4); off += 4;
    size_t n = (size_t)e.w * e.h;
    for (uint32_t c = 0; c < e.nc; ++c) { e.in[c] = (const float*)(cb+off); off += 4*n; }

    printf("expected: %ux%u, %u comps, %u res, mct=%u\n", e.w, e.h, e.nc, e.numres, e.mct);

    /* Decode the .j2c through Tier-1 only via the new API. */
    opj_set_t1_output_callback(on_t1, &e, OPJ_TRUE);
    opj_stream_t* stream = opj_stream_create_default_file_stream(j2c, 1);
    if (!stream) { fprintf(stderr, "stream fail\n"); return 1; }
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_info_handler(codec, quiet, NULL);
    opj_set_warning_handler(codec, quiet, NULL);
    opj_set_error_handler(codec, quiet, NULL);
    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    if (!opj_setup_decoder(codec, &params)) { fprintf(stderr, "setup fail\n"); return 1; }
    opj_image_t* image = NULL;
    if (!opj_read_header(stream, codec, &image)) { fprintf(stderr, "header fail\n"); return 1; }
    if (!opj_decode(codec, stream, image)) { fprintf(stderr, "decode fail\n"); return 1; }
    opj_end_decompress(codec, stream);
    opj_image_destroy(image);
    opj_destroy_codec(codec);
    opj_stream_destroy(stream);
    opj_set_t1_output_callback(NULL, NULL, OPJ_FALSE);

    printf("callback fired: %d   mismatches: %d\n", e.fired, e.mismatches);
    if (e.fired == 1 && e.mismatches == 0) {
        printf("PASS: decode-to-T1 output matches the proven back-end input exactly\n");
        return 0;
    }
    printf("FAIL\n");
    return 2;
}
