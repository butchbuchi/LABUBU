// Equivalent execution backend; no heuristic or seed changes.
#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>
struct PythonMT {
 uint32_t state[624]; int index;
 PythonMT(const uint32_t* s,int i):index(i){std::copy(s,s+624,state);}
 uint32_t next(){
  if(index>=624){
   for(int i=0;i<624;++i){uint32_t y=(state[i]&0x80000000U)|(state[(i+1)%624]&0x7fffffffU);state[i]=state[(i+397)%624]^(y>>1)^((y&1)?0x9908b0dfU:0U);}index=0;
  }
  uint32_t y=state[index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;y^=y>>18;return y;
 }
 int choice(int n){int k=n==1?1:2;uint32_t r;do{r=next()>>(32-k);}while(r>=uint32_t(n));return int(r);}
};
extern "C" int labubu_run(int T,int Q,int M,int iterations,int interval,int max_repair,
 const int* g,const int* initial,const int* caps,const uint32_t* mt,int mtindex,
 int* output,int64_t* cp_cost,int* cp_iter){
 try{
  std::vector<int> map(initial,initial+T*Q),best=map,proposal(Q),candidate(Q),counts(M),excess(M);
  PythonMT rng(mt,mtindex);int cp=0;int64_t bestcost=std::numeric_limits<int64_t>::max();
  auto cost=[&](){int64_t total=0;for(int t=0;t<T;++t)for(int q=0;q<Q;++q){int p=g[t*Q+q];if(p>q)total+=map[t*Q+q]!=map[t*Q+p];if(t)total+=map[t*Q+q]!=map[(t-1)*Q+q];}return total;};
  auto checkpoint=[&](int marker){int64_t c=cost();cp_cost[cp]=c;cp_iter[cp++]=marker;if(c<bestcost){bestcost=c;best=map;}};
  auto valid=[&](){std::fill(counts.begin(),counts.end(),0);for(int v:candidate)++counts[v];bool ok=true;for(int i=0;i<M;++i){excess[i]=std::max(0,counts[i]-caps[i]);if(excess[i])ok=false;}return ok;};
  for(int epoch=0;epoch<iterations;++epoch){
   bool bp=100<=epoch && epoch<iterations-100 && epoch%interval==0;
   if(bp)checkpoint(epoch+1);
   for(int t=0;t<T;++t){
    int* cur=map.data()+t*Q;
    for(int q=0;q<Q;++q){
     int c[3],n=0;if(t+1<T)c[n++]=map[(t+1)*Q+q];if(t)c[n++]=map[(t-1)*Q+q];int p=g[t*Q+q];if(p!=-1)c[n++]=cur[p];if(!n)c[n++]=cur[q];
     if(!bp){if(n==2 && c[0]==c[1])n=1;else if(n==3){if(c[0]==c[1]||c[0]==c[2])n=1;else if(c[1]==c[2]){c[0]=c[1];n=1;}}}
     proposal[q]=c[rng.choice(n)];
    }
    candidate=proposal;bool accepted=false;
    for(int r=0;r<max_repair;++r){
     if(valid()){accepted=true;break;}
     for(int q=0;q<Q;++q){int target=candidate[q];if(excess[target]>0 && cur[q]!=target){--excess[target];candidate[q]=cur[q];}}
     if(valid()){accepted=true;break;}
    }
    if(accepted)std::copy(candidate.begin(),candidate.end(),cur);
   }
  }
  checkpoint(-iterations);std::copy(best.begin(),best.end(),output);return cp;
 }catch(...){return -1;}
}
