/* Minimal OpenJPEG decode harness for sampling-profiler per-stage attribution.
   Decodes one J2K frame N times in-process so `sample` can attribute time to
   opj_t1_* / opj_dwt_* / opj_mct_* / opj_t2_* symbols. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openjpeg.h"

static void quiet(const char *msg, void *d){ (void)msg;(void)d; }

int main(int argc, char **argv){
    if(argc < 2){ fprintf(stderr,"usage: %s file.j2c [iters] [threads]\n",argv[0]); return 1; }
    const char *path = argv[1];
    int iters = argc>2 ? atoi(argv[2]) : 200;
    int threads = argc>3 ? atoi(argv[3]) : 1;

    for(int it=0; it<iters; ++it){
        opj_stream_t *stream = opj_stream_create_default_file_stream(path, 1);
        if(!stream){ fprintf(stderr,"stream fail\n"); return 1; }
        opj_codec_t *codec = opj_create_decompress(OPJ_CODEC_J2K);
        opj_set_info_handler(codec, quiet, NULL);
        opj_set_warning_handler(codec, quiet, NULL);
        opj_set_error_handler(codec, quiet, NULL);
        opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
        if(!opj_setup_decoder(codec, &params)){ fprintf(stderr,"setup fail\n"); return 1; }
        if(threads>1) opj_codec_set_threads(codec, threads);
        opj_image_t *image = NULL;
        if(!opj_read_header(stream, codec, &image)){ fprintf(stderr,"header fail\n"); return 1; }
        if(!opj_decode(codec, stream, image)){ fprintf(stderr,"decode fail\n"); return 1; }
        if(!opj_end_decompress(codec, stream)){ fprintf(stderr,"end fail\n"); return 1; }
        opj_image_destroy(image);
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
    }
    printf("done %d iters\n", iters);
    return 0;
}
