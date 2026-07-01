// 챕터9 bf16 GDN state de-risk (CPU): recurrence 누적 발산 측정 (정확성은 결정적 — 격리 OK).
//   gdn_step_batch 수식 그대로 N스텝 재생. fp32 state(golden) vs bf16 state(저장 bf16, 계산 fp32=mixed).
//   측정: state relerr 성장(cascade?), output max|Δ|, output argmax(=선택 v) 발산. g_t=exp(G) decay가 관건.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
static inline float bf16r(float f){ uint32_t u; memcpy(&u,&f,4); u=(u+0x8000u)&0xFFFF0000u; memcpy(&f,&u,4); return f; } // round-to-nearest bf16
int main(int argc,char**argv){
    int D=argc>1?atoi(argv[1]):128, N=argc>2?atoi(argv[2]):512;
    double Gmean=argc>3?atof(argv[3]):-0.05;   // G 평균(g_t=exp(G) decay; 작을수록 느린 forget=누적↑). 기본 g_t≈0.95
    unsigned seed=argc>4?atoi(argv[4]):42;
    printf("[gdn bf16 de-risk] D=%d N=%d Gmean=%.3f(g_t≈%.3f) seed=%u\n",D,N,Gmean,exp(Gmean),seed);
    std::mt19937 rng(seed); std::normal_distribution<float> nrm(0,1); std::uniform_real_distribution<float> ur(-0.02,0.02);
    // state: S[k][v]. fp32 golden Sf, bf16-stored Sb(계산은 fp32, 저장 시 bf16 반올림).
    std::vector<float> Sf(D*D,0.f), Sb(D*D,0.f);
    double maxrel=0, maxout=0; int argmax_mism=0, out_checks=0;
    std::vector<float> q(D),k(D),v(D),qs(D),ks(D);
    for(int t=0;t<N;t++){
        for(int i=0;i<D;i++){q[i]=nrm(rng);k[i]=nrm(rng);v[i]=nrm(rng);}
        float G=Gmean+ur(rng), beta=0.5f+0.4f*ur(rng)/0.02f; // beta~[0.1,0.9]
        // rmsnorm q,k (커널: qs=q*rsqrt(mean(q^2))*invsD, ks=k*rsqrt(mean(k^2)))  — 커널은 sum(q^2) 사용
        float sq=0,sk=0; for(int i=0;i<D;i++){sq+=q[i]*q[i];sk+=k[i]*k[i];}
        float invsD=1.f/sqrtf((float)D);
        for(int i=0;i<D;i++){qs[i]=q[i]/sqrtf(sq+1e-6f)*invsD; ks[i]=k[i]/sqrtf(sk+1e-6f);}
        float g_t=expf(G);
        // ── fp32 golden step ──
        std::vector<float> outf(D,0.f);
        for(int vv=0;vv<D;vv++){
            float kv=0; for(int kk=0;kk<D;kk++){ Sf[kk*D+vv]*=g_t; kv+=Sf[kk*D+vv]*ks[kk]; }
            float delta=(v[vv]-kv)*beta, o=0;
            for(int kk=0;kk<D;kk++){ Sf[kk*D+vv]+=ks[kk]*delta; o+=Sf[kk*D+vv]*qs[kk]; }
            outf[vv]=o;
        }
        // ── bf16-state step (저장 bf16, 계산 fp32) ──
        std::vector<float> outb(D,0.f);
        for(int vv=0;vv<D;vv++){
            float kv=0; for(int kk=0;kk<D;kk++){ float s=Sb[kk*D+vv]*g_t; kv+=s*ks[kk]; Sb[kk*D+vv]=s; }
            float delta=(v[vv]-kv)*beta, o=0;
            for(int kk=0;kk<D;kk++){ float s=Sb[kk*D+vv]+ks[kk]*delta; o+=s*qs[kk]; Sb[kk*D+vv]=bf16r(s); } // 저장 시 bf16 반올림
            outb[vv]=o;
        }
        // 측정: state relerr, output diff, argmax(v 선택) 일치
        double sn=0,sd=0; for(int i=0;i<D*D;i++){ double d=Sb[i]-Sf[i]; sn+=d*d; sd+=(double)Sf[i]*Sf[i]; }
        double rel=sqrt(sn/(sd+1e-12)); if(rel>maxrel)maxrel=rel;
        double od=0; int af=0,ab=0; for(int i=0;i<D;i++){ double d=fabs(outb[i]-outf[i]); if(d>od)od=d; if(outf[i]>outf[af])af=i; if(outb[i]>outb[ab])ab=i; }
        if(od>maxout)maxout=od;
        out_checks++; if(af!=ab)argmax_mism++;
        if(t<5||t==N-1||t%128==0) printf("  step %4d: state relerr %.3e  out max|Δ| %.3e  argmax(v) %s\n",t,rel,od,af==ab?"OK":"MISM");
    }
    printf("== 종합: max state relerr %.3e, max out|Δ| %.3e, out argmax mism %d/%d ==\n",maxrel,maxout,argmax_mism,out_checks);
    printf("== 판정: relerr 성장(cascade)? decay(g_t<1)로 plateau? argmax 유지? ==\n");
    return 0;
}
