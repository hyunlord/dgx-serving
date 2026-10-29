#ifndef DGX_GEMMA4_DECODE_GRAPH_CUH
#define DGX_GEMMA4_DECODE_GRAPH_CUH

static void g4g_require(bool ok,const char *message){
    if(!ok){fprintf(stderr,"G4_GRAPH_ERROR %s\n",message);exit(1);}
}

// Lexicographic (value descending, index ascending) reduction matches the
// first maximum selected by std::max_element, including equal signed zeros.
static __global__ void g4g_argmax(const float *x,int n,int *feedback,int *result){
    __shared__ float values[256];
    __shared__ int indices[256],bad[256];
    int t=threadIdx.x, index=INT_MAX, invalid=0;
    float value=-INFINITY;
    for(int i=t;i<n;i+=256){
        float v=x[i]; invalid|=!isfinite(v);
        if(v>value || (v==value && i<index)){value=v;index=i;}
    }
    values[t]=value;indices[t]=index;bad[t]=invalid;__syncthreads();
    for(int stride=128;stride;stride/=2){
        if(t<stride){
            float v=values[t+stride];int i=indices[t+stride];
            if(v>values[t] || (v==values[t] && i<indices[t])){values[t]=v;indices[t]=i;}
            bad[t]|=bad[t+stride];
        }
        __syncthreads();
    }
    if(t==0){*result=bad[0]?-2:indices[0];*feedback=bad[0]?0:indices[0];}
}
static __global__ void g4g_snapshot(const float *src,float *dst,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=src[i];
}

int gemma4_graph_argmax_test(){
    std::vector<float> h(1025,-3.f); float *d=nullptr;int *out=nullptr;
    DK(cudaMalloc(&d,h.size()*sizeof(float)));DK(cudaMalloc(&out,2*sizeof(int)));
    for(int trial=0;trial<4;++trial){
        std::fill(h.begin(),h.end(),-3.f);
        if(trial==0){h[1]=7;h[256]=7;}
        if(trial==1){h[0]=-0.f;h[257]=0.f;}
        if(trial==2){h[1024]=8;h[511]=8;}
        if(trial==3)h[17]=NAN;
        DK(cudaMemcpy(d,h.data(),h.size()*sizeof(float),cudaMemcpyHostToDevice));
        g4g_argmax<<<1,256>>>(d,(int)h.size(),out,out+1);
        int got;DK(cudaMemcpy(&got,out+1,sizeof(int),cudaMemcpyDeviceToHost));
        int expected=trial==3?-2:(int)(std::max_element(h.begin(),h.end())-h.begin());
        g4g_require(got==expected,"argmax synthetic comparison");
    }
    DK(cudaFree(d));DK(cudaFree(out));return 0;
}

struct G4GraphPatch {
    cudaGraphNode_t node=nullptr;
    cudaKernelNodeParams params{};
    std::array<uint64_t,16> storage{};
    std::array<void*,16> args{};
    int kind=0,layer=-1,hd=0,kvdim=0;
    bool split=false;   // kind 4 is the split-KV pass 1 (grid width follows the position)
    template<class T> T get(int i) const {T v;memcpy(&v,&storage[i],sizeof(T));return v;}
    template<class T> void set(int i,T v){storage[i]=0;memcpy(&storage[i],&v,sizeof(T));}
    void bind(){for(size_t i=0;i<args.size();++i)args[i]=&storage[i];params.kernelParams=args.data();}
};

#define G4_MAP_REQUIRE(ok,message) do { if(!(ok)){ fprintf(stderr,"G4_GRAPH_MAP_FALLBACK %s\n",message); return {}; } } while(0)

// A captured stream is a dependency chain. Traverse edges, never GetNodes order.
static std::vector<cudaGraphNode_t> g4g_chain(cudaGraph_t graph){
    size_t n=0;DK(cudaGraphGetNodes(graph,nullptr,&n));
    std::vector<cudaGraphNode_t> all(n),chain;
    DK(cudaGraphGetNodes(graph,all.data(),&n));
    cudaGraphNode_t root=nullptr;
    for(auto node:all){size_t deps=0;DK(cudaGraphNodeGetDependencies(node,nullptr,nullptr,&deps));
        G4_MAP_REQUIRE(deps<=1,"captured graph is not a chain");
        if(!deps){G4_MAP_REQUIRE(root==nullptr,"multiple graph roots");root=node;}}
    for(auto node=root;node;){chain.push_back(node);size_t count=0;
        DK(cudaGraphNodeGetDependentNodes(node,nullptr,nullptr,&count));G4_MAP_REQUIRE(count<=1,"branched graph");
        cudaGraphNode_t next=nullptr;if(count)DK(cudaGraphNodeGetDependentNodes(node,&next,nullptr,&count));node=next;}
    G4_MAP_REQUIRE(chain.size()==n,"graph chain coverage");return chain;
}

// Split-KV decode attention kernels that carry a position argument (kernels/attention_decode_split.cuh).
static bool g4g_is_split_p1(const void *f){
    return f==(const void*)attn_split_p1_kernel<256,2> || f==(const void*)attn_split_p1_kernel<512,16> ||
           f==(const void*)attn_split_p1_kernel<256,1> || f==(const void*)attn_split_p1_kernel<256,4> ||
           f==(const void*)attn_split_p1_kernel<512,2>;
}
static bool g4g_is_split_combine(const void *f){
    return f==(const void*)attn_split_combine_kernel<256> || f==(const void*)attn_split_combine_kernel<512>;
}

static std::vector<G4GraphPatch> g4g_map(cudaGraph_t graph,const ModelConfig &c,
        const G4Weights &w,const G4DecodeState &st,int base,const float *cf,const float *sf,
        const float *cs,const float *ss){
    std::vector<G4GraphPatch> out;
    std::vector<std::array<int,5>> roles(c.num_layers);
    int rope_count=0,slots=0,combines=0,last_split_layer=-1;
    for(auto node:g4g_chain(graph)){
        cudaGraphNodeType type;DK(cudaGraphNodeGetType(node,&type));
        G4_MAP_REQUIRE(type==cudaGraphNodeTypeKernel,"unexpected non-kernel graph node");
        cudaKernelNodeParams p{};DK(cudaGraphKernelNodeGetParams(node,&p));
        G4GraphPatch patch;patch.node=node;patch.params=p;
        std::vector<size_t> sizes;
        if(p.func==(void*)rope_kernel){patch.kind=1;sizes={8,8,8,4,4,4};}
        else if(p.func==(void*)rmsnorm_kernel){patch.kind=2;sizes={8,8,8,4,4};}
        else if(p.func==(void*)rmsnorm_ns_kernel){patch.kind=3;sizes={8,8,4,4};}
        else if(p.func==(void*)attn_decode_kernel){patch.kind=4;sizes={8,8,8,8,4,4,4,4,4,4,4};}
        else if(g4g_is_split_p1(p.func)){patch.kind=4;patch.split=true;sizes={8,8,8,8,8,8,4,4,4,4,8,8,4,4,4,4};}
        else if(g4g_is_split_combine(p.func)){patch.kind=9;sizes={8,8,8,8,4,4,4,8,4,4,4};}
        else if(p.func==(void*)g4g_argmax){patch.kind=5;sizes={8,4,8,8};}
        else if(p.func==(void*)g4g_snapshot){patch.kind=6;sizes={8,8,4};}
        else continue;
        for(size_t i=0;i<sizes.size();++i)memcpy(&patch.storage[i],p.kernelParams[i],sizes[i]);
        int l=-1;
        if(patch.kind==9){
            G4_MAP_REQUIRE(last_split_layer>=0,"combine without split pass 1");
            G4_MAP_REQUIRE(patch.get<int>(5)==base && patch.get<const int*>(7)==nullptr,"combine position signature");
            patch.layer=last_split_layer;last_split_layer=-1;++combines;out.push_back(patch);continue;
        }
        if(patch.kind==1){l=rope_count/2;G4_MAP_REQUIRE(l<c.num_layers,"extra rope node");++rope_count;}
        else if(patch.kind<5){
            const float *ptr=patch.get<const float*>(patch.kind==2?2:1);
            for(int k=0;k<c.num_layers;++k){
                int hd=w.layers[k].is_full?(c.global_head_dim?c.global_head_dim:c.head_dim):c.head_dim;
                int nkv=w.layers[k].is_full?(c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads):c.num_kv_heads;
                const float *expected=patch.kind==3?kv_at(st.Vc[k],(size_t)base*hd*nkv):
                    patch.kind==4?st.Kc[k]:kv_at(st.Kc[k],(size_t)base*hd*nkv);
                if(ptr==expected){G4_MAP_REQUIRE(l==-1,"ambiguous KV node");l=k;}
            }
            if(l<0){G4_MAP_REQUIRE(patch.kind!=4,"unmapped attention");continue;}
        }
        if(l>=0){
            patch.layer=l;bool full=w.layers[l].is_full;
            patch.hd=full?(c.global_head_dim?c.global_head_dim:c.head_dim):c.head_dim;
            int nkv=full?(c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads):c.num_kv_heads;
            patch.kvdim=patch.hd*nkv;
            if(patch.kind==1){
                const float *co=full?cf:cs,*si=full?sf:ss;
                G4_MAP_REQUIRE(patch.get<const float*>(1)==co+(size_t)base*patch.hd &&
                    patch.get<const float*>(2)==si+(size_t)base*patch.hd,"rope source mismatch");
                bool key=(rope_count%2)==0;
                G4_MAP_REQUIRE(patch.get<int>(3)==1 && patch.get<int>(4)==(key?nkv:c.num_q_heads) &&
                    patch.get<int>(5)==patch.hd,"rope signature mismatch");
                if(key){G4_MAP_REQUIRE(patch.get<float*>(0)==kv_at(st.Kc[l],(size_t)base*patch.kvdim),"K rope output");patch.kind=7;}
                ++roles[l][key?1:0];slots+=key?3:2;
            } else {
                if(patch.kind==4 && patch.split){
                    G4_MAP_REQUIRE(patch.get<int>(8)==base && patch.get<const int*>(10)==nullptr && patch.get<long>(11)==0 &&
                                   patch.params.gridDim.z==1,"split position signature");
                    last_split_layer=l;
                }
                ++roles[l][patch.kind];++slots;}
        }
        out.push_back(patch);
    }
    for(auto r:roles)for(int count:r)G4_MAP_REQUIRE(count==1,"layer position-node role coverage");
    G4_MAP_REQUIRE(rope_count==2*c.num_layers && slots==8*c.num_layers,"position argument coverage");
    G4_MAP_REQUIRE(combines==0 || combines==c.num_layers,"split combine coverage");
    return out;
}

#undef G4_MAP_REQUIRE

int gemma4_generate(const ModelConfig &c,const G4Weights &w,G4DecodeState &st,
        int first_token,int max_steps,int eos_token,const float *cf,const float *sf,
        const float *cs,const float *ss,int use_q4,G4GenerationCallback callback,
        void *opaque,G4GenerationStats *result_stats,unsigned flags){
    G4GenerationStats stats;
    g4g_require(max_steps>=0 && first_token>=0 && first_token<c.vocab_size,"generation input");
    g4g_require(st.n_past>=0 && max_steps<=st.max_len-st.n_past,"generation capacity");
    if(!max_steps){if(result_stats)*result_stats=stats;return 0;}
    int start=st.n_past,V=c.vocab_size;
    // Default (unset) = v2 overlap; G4_CUDA_GRAPH=0 = eager, 1 = per-step v1 capture.
    const char *mode=getenv("G4_CUDA_GRAPH");bool v2=!mode || strcmp(mode,"2")==0;
    // v2 = overlap by default (fastest measured R13 configuration); node updates are opt-in.
    auto option=[](const char *name,bool fallback){const char *v=getenv(name);return v?strcmp(v,"0")!=0:fallback;};
    bool overlap=v2 && option("G4_GRAPH_OVERLAP",true),direct=v2 && option("G4_GRAPH_NODE_UPDATE",false);
    const bool fixed=(flags&G4_GENERATE_FIXED_INPUT)!=0;
    float *logits=nullptr,*snapshots[2]={};int *token=nullptr,*out=nullptr,*host=nullptr,*sink=nullptr;
    DK(cudaMalloc(&logits,(size_t)V*4));DK(cudaMalloc(&token,sizeof(int)));DK(cudaMalloc(&sink,sizeof(int)));
    DK(cudaMalloc(&out,2*sizeof(int)));DK(cudaMallocHost(&host,2*sizeof(int)));
    DK(cudaMemcpy(token,&first_token,sizeof(int),cudaMemcpyHostToDevice));
    std::vector<float> host_logits(V);
    if(!v2){
        const G4GenerationStats before=g4d_graph_totals;
        for(int i=0;i<max_steps;++i){
            gemma4_decode_step(c,w,st,token,1,cf,sf,cs,ss,logits,1,use_q4,1,0);
            DK(cudaMemcpy(host_logits.data(),logits,(size_t)V*4,cudaMemcpyDeviceToHost));
            for(float v:host_logits)g4g_require(std::isfinite(v),"nonfinite logits");
            int next=(int)(std::max_element(host_logits.begin(),host_logits.end())-host_logits.begin());
            ++stats.executed_steps;++stats.accepted_steps;
            const int stop=callback?callback(logits,next,i,opaque):0;
            if(next==eos_token || stop)break;
            if(!fixed)DK(cudaMemcpy(token,&next,sizeof(int),cudaMemcpyHostToDevice));
        }
        stats.planned_captures=g4d_graph_totals.planned_captures-before.planned_captures;
        stats.graph_updates=g4d_graph_totals.graph_updates-before.graph_updates;
        stats.update_failures=g4d_graph_totals.update_failures-before.update_failures;
        stats.instantiations=g4d_graph_totals.instantiations-before.instantiations;
        stats.reinstantiations=g4d_graph_totals.reinstantiations-before.reinstantiations;
        stats.capture_us=g4d_graph_totals.capture_us-before.capture_us;
        stats.update_us=g4d_graph_totals.update_us-before.update_us;
    } else {
        using Clock=std::chrono::steady_clock;
        auto us=[](Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();};
        cudaStream_t execute=nullptr,capture=nullptr;cudaEvent_t done[2]={};
        cudaGraphExec_t exec[2]={};cudaGraph_t retained=nullptr;
        DK(cudaStreamCreateWithFlags(&execute,cudaStreamNonBlocking));
        DK(cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking));
        for(int i=0;i<2;++i){DK(cudaEventCreateWithFlags(&done[i],cudaEventDisableTiming));DK(cudaMalloc(&snapshots[i],(size_t)V*4));}
        G4PreparedStep enqueue;
        gemma4_decode_impl(c,w,st,token,1,cf,sf,cs,ss,logits,0,use_q4,1,0,nullptr,false,&enqueue);
        DK(cudaDeviceSynchronize());
        std::vector<G4GraphPatch> patches;
        bool mapping_fallback=false;
        auto record=[&](int position,int slot){
            auto begin=Clock::now();cudaGraph_t graph=nullptr;
            DK(cudaStreamBeginCapture(capture,cudaStreamCaptureModeGlobal));
            DK(enqueue(capture,position,token,logits));
            if(overlap){
                g4g_snapshot<<<(V+255)/256,256,0,capture>>>(logits,snapshots[slot],V);
                g4g_argmax<<<1,256,0,capture>>>(logits,V,fixed?sink:token,out+slot);
            }
            DK(cudaGetLastError());DK(cudaStreamEndCapture(capture,&graph));
            if(mapping_fallback)++stats.fallback_recaptures;else ++stats.planned_captures;stats.capture_us+=us(begin,Clock::now());return graph;
        };
        auto prepare=[&](int step){
            int slot=overlap?step%2:0,position=start+step;
            if(direct){
                if(!retained){
                    retained=record(position,0);patches=g4g_map(retained,c,w,st,position,cf,sf,cs,ss);
                    if(patches.empty()){
                        // A valid capture remains executable even when its
                        // signature is unknown to the direct-update mapper.
                        direct=false;mapping_fallback=true;++stats.mapping_fallbacks;
                        DK(cudaGraphInstantiate(&exec[slot],retained,nullptr,nullptr,0));++stats.instantiations;
                        DK(cudaGraphDestroy(retained));retained=nullptr;
                        return;
                    }
                    for(int j=0;j<(overlap?2:1);++j){DK(cudaGraphInstantiate(&exec[j],retained,nullptr,nullptr,0));++stats.instantiations;}
                }
                auto begin=Clock::now();
                for(auto &p:patches){
                    int l=p.layer;
                    if(p.kind==1 || p.kind==7){bool full=w.layers[l].is_full;
                        p.set(1,(full?cf:cs)+(size_t)position*p.hd);p.set(2,(full?sf:ss)+(size_t)position*p.hd);
                        if(p.kind==7)p.set(0,st.Kc[l]+(size_t)position*p.kvdim);
                    } else if(p.kind==2)p.set(2,st.Kc[l]+(size_t)position*p.kvdim);
                    else if(p.kind==3)p.set(1,st.Vc[l]+(size_t)position*p.kvdim);
                    else if(p.kind==4 && p.split){p.set(8,position);p.params.gridDim.x=position/attn_split_chunk()+1;}
                    else if(p.kind==4){p.set(8,position);p.params.sharedMemBytes=(p.hd+position+1)*sizeof(float);}
                    else if(p.kind==9)p.set(5,position);
                    else if(p.kind==5)p.set(3,out+slot);
                    else if(p.kind==6)p.set(1,snapshots[slot]);
                    p.bind();cudaError_t status=cudaGraphExecKernelNodeSetParams(exec[slot],p.node,&p.params);
                    if(status!=cudaSuccess)++stats.setparams_failures;DK(status);++stats.node_updates;
                }
                stats.update_us+=us(begin,Clock::now());
            } else {
                cudaGraph_t graph=record(position,slot);auto begin=Clock::now();
                if(exec[slot]){
                    cudaGraphExecUpdateResultInfo info{};cudaError_t status=cudaGraphExecUpdate(exec[slot],graph,&info);
                    if(status==cudaSuccess && info.result==cudaGraphExecUpdateSuccess)++stats.graph_updates;
                    if(status!=cudaSuccess || info.result!=cudaGraphExecUpdateSuccess){
                        ++stats.update_failures;
                        if(status!=cudaSuccess && status!=cudaErrorGraphExecUpdateFailure)DK(status);
                        cudaError_t last=cudaGetLastError();
                        if(last!=cudaSuccess && last!=cudaErrorGraphExecUpdateFailure)DK(last);
                        DK(cudaGraphExecDestroy(exec[slot]));exec[slot]=nullptr;++stats.reinstantiations;
                    }
                }
                if(!exec[slot]){DK(cudaGraphInstantiate(&exec[slot],graph,nullptr,nullptr,0));++stats.instantiations;}
                DK(cudaGraphDestroy(graph));stats.update_us+=us(begin,Clock::now());
            }
        };
        auto launch=[&](int step){int slot=overlap?step%2:0;
            DK(cudaGraphLaunch(exec[slot],execute));
            if(overlap)DK(cudaMemcpyAsync(host+slot,out+slot,sizeof(int),cudaMemcpyDeviceToHost,execute));
            DK(cudaEventRecord(done[slot],execute));++stats.executed_steps;
        };
        prepare(0);launch(0);
        for(int i=0;i<max_steps;++i){
            int slot=overlap?i%2:0;
            // At most one successor is in flight; the previous callback has
            // consumed this ring slot before it can be reused two steps later.
            if(overlap && i+1<max_steps){prepare(i+1);launch(i+1);}
            DK(cudaEventSynchronize(done[slot]));
            int next;
            if(overlap){next=host[slot];g4g_require(next>=0 && next<V,"nonfinite device argmax");}
            else {
                DK(cudaMemcpy(host_logits.data(),logits,(size_t)V*4,cudaMemcpyDeviceToHost));
                for(float v:host_logits)g4g_require(std::isfinite(v),"nonfinite logits");
                next=(int)(std::max_element(host_logits.begin(),host_logits.end())-host_logits.begin());
            }
            ++stats.accepted_steps;st.n_past=start+(int)stats.accepted_steps;
            const int stop=callback?callback(overlap?snapshots[slot]:logits,next,i,opaque):0;
            if(next==eos_token || stop)break;
            if(!overlap && i+1<max_steps){
                if(!fixed)DK(cudaMemcpyAsync(token,&next,sizeof(int),cudaMemcpyHostToDevice,execute));
                DK(cudaStreamSynchronize(execute));prepare(i+1);launch(i+1);
            }
        }
        DK(cudaStreamSynchronize(execute));
        stats.speculative_steps=stats.executed_steps-stats.accepted_steps;
        st.n_past=start+(int)stats.accepted_steps;
        for(int j=0;j<2;++j){if(exec[j])DK(cudaGraphExecDestroy(exec[j]));DK(cudaEventDestroy(done[j]));DK(cudaFree(snapshots[j]));}
        if(retained)DK(cudaGraphDestroy(retained));DK(cudaStreamDestroy(capture));DK(cudaStreamDestroy(execute));
    }
    DK(cudaFree(logits));DK(cudaFree(token));DK(cudaFree(sink));DK(cudaFree(out));DK(cudaFreeHost(host));
    if(result_stats)*result_stats=stats;
    if(const char *v=getenv("G4_CUDA_GRAPH_STATS"))if(strcmp(v,"0")!=0)
        fprintf(stderr,"G4_GENERATION_STATS planned_captures=%llu fallback_recaptures=%llu node_updates=%llu graph_updates=%llu mapping_fallbacks=%llu instantiations=%llu reinstantiations=%llu update_failures=%llu setparams_failures=%llu executed_steps=%llu accepted_steps=%llu speculative_steps=%llu capture_us=%.6f update_us=%.6f\n",stats.planned_captures,stats.fallback_recaptures,stats.node_updates,stats.graph_updates,stats.mapping_fallbacks,stats.instantiations,stats.reinstantiations,stats.update_failures,stats.setparams_failures,stats.executed_steps,stats.accepted_steps,stats.speculative_steps,stats.capture_us,stats.update_us);
    return (int)stats.accepted_steps;
}
#endif
