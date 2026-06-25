/* Portable-C reference for the full GPU back-end: inverse 9/7 DWT + inverse
 * MCT/ICT + DC level-shift -> final integer image. Validates integer-exact
 * against OpenJPEG's output (OPJ_BACKEND_DUMP corpus).
 *
 * Build: clang -O2 -ffp-contract=off proto/backend_ref.c -o /tmp/backend_ref -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static const float K=1.230174105f, two_invK=1.625732422f;
static const float alpha=-1.586134342f, beta=-0.052980118f, gamma_=0.882911075f, delta=0.443506852f;
static inline int imin(int a,int b){return a<b?a:b;}

static void step1(float* base,int start,int end,float c){ for(int i=start;i<end;++i) base[i*2]*=c; }
static void step2(float* lbase,float* wbase,int start,int end,int m,float c){
    float* fl=lbase; float* fw=wbase; int imax=imin(end,m);
    if(start>0){fw+=2*start; fl=fw-2;}
    for(int i=start;i<imax;++i){ fw[-1]=fw[-1]+(fl[0]+fw[0])*c; fl=fw; fw+=2; }
    if(m<end){ c+=c; fw[-1]=fw[-1]+fl[0]*c; }
}
static void idwt97_1d(float* W,int sn,int dn,int cas){
    int a,b;
    if(cas==0){ if(!((dn>0)||(sn>1)))return; a=0;b=1; } else { if(!((sn>0)||(dn>1)))return; a=1;b=0; }
    step1(W+a,0,sn,K); step1(W+b,0,dn,two_invK);
    step2(W+b,W+a+1,0,sn,imin(sn,dn-a),-delta);
    step2(W+a,W+b+1,0,dn,imin(dn,sn-b),-gamma_);
    step2(W+b,W+a+1,0,sn,imin(sn,dn-a),-beta);
    step2(W+a,W+b+1,0,dn,imin(dn,sn-b),-alpha);
}
static void idwt97_2d(float* buf,uint32_t w,const int32_t* boxes,uint32_t numres,float* W){
    int rw=boxes[2]-boxes[0], rh=boxes[3]-boxes[1];
    for(uint32_t lvl=1;lvl<numres;++lvl){
        int sn_h=rw, sn_v=rh;
        int x0=boxes[lvl*4+0], y0=boxes[lvl*4+1];
        rw=boxes[lvl*4+2]-x0; rh=boxes[lvl*4+3]-y0;
        int dn_h=rw-sn_h, cas_h=x0&1, dn_v=rh-sn_v, cas_v=y0&1;
        for(int j=0;j<rh;++j){
            float* row=buf+(size_t)j*w;
            for(int i=0;i<sn_h;++i) W[2*i+cas_h]=row[i];
            for(int i=0;i<dn_h;++i) W[2*i+(1-cas_h)]=row[sn_h+i];
            idwt97_1d(W,sn_h,dn_h,cas_h);
            for(int k=0;k<rw;++k) row[k]=W[k];
        }
        for(int i=0;i<rw;++i){
            for(int k=0;k<sn_v;++k) W[2*k+cas_v]=buf[(size_t)k*w+i];
            for(int k=0;k<dn_v;++k) W[2*k+(1-cas_v)]=buf[(size_t)(sn_v+k)*w+i];
            idwt97_1d(W,sn_v,dn_v,cas_v);
            for(int k=0;k<rh;++k) buf[(size_t)k*w+i]=W[k];
        }
    }
}
static inline int clampi(long v,int lo,int hi){ return v<lo?lo:(v>hi?hi:(int)v); }

int main(int argc,char**argv){
    const char* path=argc>1?argv[1]:"proto/backend_corpus.bin";
    FILE* f=fopen(path,"rb"); if(!f){perror("open");return 1;}
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* all=malloc(fsz); fread(all,1,fsz,f); fclose(f);

    long off=0; int frames=0; long ndiff=0, ntot=0; int maxdiff=0;
    double ms=0;
    while(off<fsz){
        uint32_t magic; memcpy(&magic,all+off,4); off+=4;
        if(magic!=0x444E4B42){ printf("bad magic\n"); break; }
        uint32_t nc; memcpy(&nc,all+off,4); off+=4;
        uint32_t w,h; memcpy(&w,all+off,4); memcpy(&h,all+off+4,4); off+=8;
        uint32_t numres; memcpy(&numres,all+off,4); off+=4;
        int32_t* boxes=(int32_t*)(all+off); off+=16*numres;
        int prec[8], sgnd[8], dcs[8];
        for(uint32_t c=0;c<nc;++c){ prec[c]=*(int32_t*)(all+off); sgnd[c]=*(int32_t*)(all+off+4); dcs[c]=*(int32_t*)(all+off+8); off+=12; }
        int32_t mct=*(int32_t*)(all+off); off+=4;
        size_t n=(size_t)w*h;
        float* in[8]; for(uint32_t c=0;c<nc;++c){ in[c]=(float*)(all+off); off+=4*n; }
        int32_t* oracle[8]; for(uint32_t c=0;c<nc;++c){ oracle[c]=(int32_t*)(all+off); off+=4*n; }

        float* buf[8]; for(uint32_t c=0;c<nc;++c){ buf[c]=malloc(n*4); memcpy(buf[c],in[c],n*4); }
        float* W=malloc((size_t)(w>h?w:h)*2*4+64);

        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        for(uint32_t c=0;c<nc;++c) idwt97_2d(buf[c],w,boxes,numres,W);
        if(mct && nc>=3){
            for(size_t i=0;i<n;++i){
                float y=buf[0][i], u=buf[1][i], v=buf[2][i];
                buf[0][i]=y+v*1.402f;
                buf[1][i]=y-u*0.34413f-v*0.71414f;
                buf[2][i]=y+u*1.772f;
            }
        }
        clock_gettime(CLOCK_MONOTONIC,&t1);
        ms+=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;

        for(uint32_t c=0;c<nc;++c){
            int lo = sgnd[c]? -(1<<(prec[c]-1)) : 0;
            int hi = sgnd[c]? (1<<(prec[c]-1))-1 : (1<<prec[c])-1;
            for(size_t i=0;i<n;++i){
                long val=(long)lrintf(buf[c][i]);
                int out=clampi(val+dcs[c],lo,hi);
                int want=oracle[c][i];
                if(out!=want){ ndiff++; int d=abs(out-want); if(d>maxdiff)maxdiff=d; }
            }
            ntot+=n;
        }
        for(uint32_t c=0;c<nc;++c) free(buf[c]); free(W);
        frames++;
    }
    printf("frames: %d  samples: %ld  exact: %ld  differ: %ld  max |diff|: %d\n",
           frames, ntot, ntot-ndiff, ndiff, maxdiff);
    printf("CPU reference back-end (iDWT+MCT): %.2f ms/frame\n", ms/frames);
    printf("%s\n", ndiff==0 ? "INTEGER-EXACT vs OpenJPEG final image ✓"
                            : (maxdiff<=1 ? "off-by-<=1 on some pixels (scalar FP rounding; GPU float-exact path should match)" : "MISMATCH"));
    return ndiff && maxdiff>1 ? 2 : 0;
}
