// Diagnostic actual-path interception. Do not also link models/gemma4_moe.cu.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <stdexcept>
#include <cuda_runtime.h>
#include "../runtime/manifest_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../models/gemma4_moe.h"
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_tiled.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/router_topk.cuh"
#include "../kernels/moe_fused.cuh"
namespace diag {
static std::map<const void*,std::string> norms,mats;
static std::string out,path;static int layer=-1;static size_t bytes=0;static FILE*index=nullptr;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> static void save(const std::string&label,const T*d,size_t n){
 if(!n || n>32*1024*1024 || bytes+n*sizeof(T)>256*1024*1024)throw std::runtime_error("capture budget exceeded");
 std::vector<T>v(n);ck(cudaMemcpy(v.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));
 for(T x:v)if(!std::isfinite((double)x))throw std::runtime_error("nonfinite "+label);
 std::string name=path+"_"+label+".bin";FILE*f=fopen((out+"/"+name).c_str(),"wbx");if(!f)throw std::runtime_error("refuse overwrite/open "+name);
 bool ok=fwrite(v.data(),sizeof(T),n,f)==n;if(fclose(f)!=0 || !ok)throw std::runtime_error("write failed");
 bytes+=n*sizeof(T);fprintf(index,"%s,%zu,%zu\n",name.c_str(),n,sizeof(T));fflush(index);
}
static std::string ln(const char*op){return "L"+std::to_string(layer)+"_"+op;}
static void norm(const float*x,const float*w,float*y,int rows,int h,float eps,cudaStream_t st=0){
 auto it=norms.find(w);if(it==norms.end())throw std::runtime_error("unmapped norm");auto name=it->second;
 if(name.find("_input_ln")!=std::string::npos)layer=std::stoi(name.substr(1));
 save(name+"_in",x,(size_t)rows*h);launch_rmsnorm(x,w,y,rows,h,eps,st);ck(cudaStreamSynchronize(st));save(name+"_out",y,(size_t)rows*h);
}
static void gemv(const int8_t*x,const float*xs,const int32_t*sum,const uint8_t*w,const void*sc,float*y,int in,int n,int s,int dtype=0,cudaStream_t st=0){
 auto it=mats.find(w);if(it==mats.end())throw std::runtime_error("unmapped q4 weight");auto name=it->second;
 save(name+"_q8",x,(size_t)s*in);save(name+"_q8scale",xs,(size_t)s*in/QBLK);save(name+"_q8sum",sum,(size_t)s*in/QBLK);
 launch_gemv_w4a8(x,xs,sum,w,sc,y,in,n,s,dtype,st);ck(cudaStreamSynchronize(st));save(name+"_out",y,(size_t)s*n);
}
static void tiled(const float*x,const uint8_t*w,const void*sc,float*y,int m,int n,int k,int dtype=0,cudaStream_t st=0){
 auto it=mats.find(w);if(it==mats.end())throw std::runtime_error("unmapped q4 weight");save(it->second+"_in",x,(size_t)m*k);
 launch_gemm_tiled_q4(x,w,sc,y,m,n,k,dtype,st);ck(cudaStreamSynchronize(st));save(it->second+"_out",y,(size_t)m*n);
}
static int quant_seq=0;
static void quant(const float*x,int8_t*q,float*sc,int32_t*sum,int in,int s,cudaStream_t st=0){
 save("quant_"+std::to_string(quant_seq++)+"_input",x,(size_t)in*s);launch_quantize_x_q8(x,q,sc,sum,in,s,st);
}
static void attn_inputs(const float*q,const float*k,const float*v,int s,int nq,int nk,int hd){save(ln("attn_q"),q,(size_t)s*nq*hd);save(ln("attn_k"),k,(size_t)s*nk*hd);save(ln("attn_v"),v,(size_t)s*nk*hd);}
static void attention(const float*q,const float*k,const float*v,float*o,int s,int nq,int nk,int hd,cudaStream_t st=0,float scale=0,int window=0){
 attn_inputs(q,k,v,s,nq,nk,hd);launch_attn(q,k,v,o,s,nq,nk,hd,st,scale,window);ck(cudaStreamSynchronize(st));save(ln("attn_out"),o,(size_t)s*nq*hd);
}
static void attention_decode(const float*q,const float*k,const float*v,float*o,int s,int nq,int nk,int hd,int base,cudaStream_t st=0,float scale=0,int window=0){
 if(base!=0)throw std::runtime_error("base must be zero");attn_inputs(q,k,v,s,nq,nk,hd);launch_attn_decode(q,k,v,o,s,nq,nk,hd,base,st,scale,window);ck(cudaStreamSynchronize(st));save(ln("attn_out"),o,(size_t)s*nq*hd);
}
}
#define launch_rmsnorm diag::norm
#define launch_gemv_w4a8 diag::gemv
#define launch_gemm_tiled_q4 diag::tiled
#define launch_quantize_x_q8 diag::quant
#define launch_attn diag::attention
#define launch_attn_decode diag::attention_decode
#include "../models/gemma4_moe.cu"
#undef launch_rmsnorm
#undef launch_gemv_w4a8
#undef launch_gemm_tiled_q4
#undef launch_quantize_x_q8
#undef launch_attn
#undef launch_attn_decode
namespace diag {
template<class T>static std::vector<T>read(const std::string&p,size_t n=0){FILE*f=fopen(p.c_str(),"rb");if(!f)throw std::runtime_error("read "+p);fseek(f,0,SEEK_END);long b=ftell(f);rewind(f);if(b<=0 || b%sizeof(T) || (n&&(size_t)b!=n*sizeof(T))){fclose(f);throw std::runtime_error("size "+p);}std::vector<T>v(b/sizeof(T));bool ok=fread(v.data(),sizeof(T),v.size(),f)==v.size();fclose(f);if(!ok)throw std::runtime_error("short read");return v;}
template<class T>static T*up(const std::vector<T>&v){T*d;ck(cudaMalloc(&d,v.size()*sizeof(T)));ck(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
static void mapping(const G4Weights&w){norms[w.final_norm]="final_norm";mats[w.lm_head.wq]="lm_head";for(int l=0;l<w.L;l++){const auto&a=w.layers[l];std::string p="L"+std::to_string(l)+"_";
#define N(f) if(a.f)norms[a.f]=p+#f
 N(input_ln);N(post_attn_ln);N(pre_ffn_ln);N(post_ffn_ln);N(q_norm);N(k_norm);N(post_ff_ln_1);N(pre_ff_ln_2);N(post_ff_ln_2);N(router_scale);
#undef N
#define M(f) if(a.f.wq)mats[a.f.wq]=p+#f
 M(q);M(k);M(v);M(o);M(gate);M(up);M(down);
#undef M
}}
static void compare(const char*label,const std::vector<float>&a,const std::vector<float>&b,int s,int v){double delta=0,den=0;int match=0;for(size_t i=0;i<a.size();i++){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite logits");delta=fmax(delta,fabs((double)a[i]-b[i]));den=fmax(den,fabs((double)b[i]));}for(int r=0;r<s;r++){int ai=0,bi=0;for(int j=1;j<v;j++){if(a[(size_t)r*v+j]>a[(size_t)r*v+ai])ai=j;if(b[(size_t)r*v+j]>b[(size_t)r*v+bi])bi=j;}match+=ai==bi;printf("ARGMAX %s row=%d actual=%d reference=%d\n",label,r,ai,bi);}printf("COMPARE %s argmax=%d/%d maxabs=%.17g normalized_maxabs=%.17g\n",label,match,s,delta,den?delta/den:(delta?INFINITY:0));}
}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("usage: diagnostic WEIGHTS GOLDEN NEW_OUT_DIR {1|2}");if(strcmp(argv[4],"1")&&strcmp(argv[4],"2"))throw std::runtime_error("uq must be 1 or 2");int uq=atoi(argv[4]);std::string wd=argv[1],gd=argv[2];diag::out=argv[3];if(std::filesystem::exists(diag::out))throw std::runtime_error("output directory must not exist");
 ModelConfig c=load_config_gemma4((wd+"/manifest.json").c_str());if(c.num_experts<=0||!c.layer_is_full||c.hidden_size<=0||c.vocab_size<=0)throw std::runtime_error("invalid MoE config");auto ids=diag::read<int>(gd+"/token_ids.bin");int s=ids.size(),hd=c.global_head_dim?c.global_head_dim:c.head_dim;if(s<1||s>8||s>hd||s>c.head_dim)throw std::runtime_error("requires 1..8 golden tokens and S<=HD");for(int t:ids)if(t<0||t>=c.vocab_size)throw std::runtime_error("token range");
 auto gold=diag::read<float>(gd+"/q4_logits.bin",(size_t)s*c.vocab_size);auto cf=diag::read<float>(gd+"/rope_full_cos.bin",(size_t)s*hd),sf=diag::read<float>(gd+"/rope_full_sin.bin",(size_t)s*hd),cs=diag::read<float>(gd+"/rope_sliding_cos.bin",(size_t)s*c.head_dim),ss=diag::read<float>(gd+"/rope_sliding_sin.bin",(size_t)s*c.head_dim);for(const auto*p:{&cf,&sf,&cs,&ss})for(float x:*p)if(!std::isfinite(x))throw std::runtime_error("nonfinite rope");
 std::filesystem::create_directory(diag::out);diag::index=fopen((diag::out+"/captures.csv").c_str(),"wx");if(!diag::index)throw std::runtime_error("index failed");fprintf(diag::index,"file,elements,itemsize\n");int*dids=diag::up(ids);float*dcf=diag::up(cf),*dsf=diag::up(sf),*dcs=diag::up(cs),*dss=diag::up(ss);G4Weights w=load_weights_gemma4(wd.c_str(),c,1,0);diag::mapping(w);float*logits;size_t n=(size_t)s*c.vocab_size;diag::ck(cudaMalloc(&logits,n*4));std::vector<float>fwd(n),dec(n);
 printf("DIAGNOSTIC uq=%d scale_dtype=%d S=%d H=%d L=%d V=%d reference=%s output=%s\n",uq,c.scale_dtype,s,c.hidden_size,c.num_layers,c.vocab_size,gd.c_str(),diag::out.c_str());diag::path="forward";diag::quant_seq=0;gemma4_moe_forward(c,w,dids,s,dcf,dsf,dcs,dss,logits,nullptr,uq,nullptr,nullptr);diag::save("logits",logits,n);diag::ck(cudaMemcpy(fwd.data(),logits,n*4,cudaMemcpyDeviceToHost));
 G4DecodeState st=alloc_g4_decode_state(c,s);reset_g4_decode_state(st);diag::path="decode";diag::layer=-1;diag::quant_seq=0;gemma4_moe_decode(c,w,st,dids,s,dcf,dsf,dcs,dss,logits,1,uq,0);diag::save("logits",logits,n);diag::ck(cudaMemcpy(dec.data(),logits,n*4,cudaMemcpyDeviceToHost));diag::compare("forward_vs_HF",fwd,gold,s,c.vocab_size);diag::compare("decode_vs_HF",dec,gold,s,c.vocab_size);diag::compare("decode_vs_forward",dec,fwd,s,c.vocab_size);
 printf("CAPTURE_COMPLETE bytes=%zu (exit 0 is capture success, not accuracy PASS)\n",diag::bytes);fclose(diag::index);free_g4_decode_state(st);free_weights_gemma4(w);cudaFree(logits);cudaFree(dids);cudaFree(dcf);cudaFree(dsf);cudaFree(dcs);cudaFree(dss);free_config_gemma4(c);return 0;
}catch(const std::exception&e){fprintf(stderr,"DIAGNOSTIC_ERROR %s\n",e.what());return 1;}}
