#ifndef DGX_SERVING_BACKEND_H
#define DGX_SERVING_BACKEND_H
#include <cstddef>

// ModelBackend 훅 (M4 서빙코어 추상화, bd23fd5 설계 → 실행). 서빙 코어(serving_core.h: admission/slot/큐/Poisson/
// 통계)는 이 인터페이스로만 모델을 접촉 — 코어에 모델 분기 없음(③ 회피). bench_serve.cu 결합 6지점을 5훅으로 뽑음.
//   backend가 모든 device 전송(d_tok/d_pos/d_logit H2D)·rope·weight·상태 버퍼를 소유 → 코어는 host 배열만 다룸(모델 무지).
//   opaque state(void*): Qwen=BatchDecodeState, Gemma4=G4BatchState. 코어는 타입 무지.
//   has_spec(): spec 서빙 사이클(draft→verify→promote) 유무. plain(Gemma4·qwen-plain)=false → decode_step=1토큰 전진.
//     [spec 확장점] spec backend는 decode_step 내부에서 사이클 캡슐화하고, 슬롯당 가변 accept를 out-배열로 반환해야 함
//     (plain은 전부 +1). 코어의 slot-advance가 그 배열을 읽게 하면 spec-core도 같은 코어에 플러그인 가능(이번 범위 밖).
struct ServingBackend {
    virtual ~ServingBackend() {}
    virtual bool   has_spec() const = 0;                       // spec 서빙 사이클 유무(plain=false)
    // per-seq state bytes(admission 예산 — GDN 항 vs KV-only). maxlen 기준. 코어의 alloc-time 가드가 호출.
    virtual size_t per_seq_bytes(int maxlen) const = 0;
    // batch state alloc(Bmax 슬롯, reset 포함)/free. 반환 opaque(코어가 그대로 되돌려줌).
    virtual void*  alloc_state(int Bmax, int maxlen) = 0;
    virtual void   free_state(void* state) = 0;
    virtual void   reset_slot(void* state, int b) = 0;         // slot 재사용 시 잔재 clear
    // 슬롯 b에 프롬프트 P토큰 chunked prefill(backend 소유 tiled 프롬프트 사용). 반환: wall(초).
    virtual double prefill_slot(void* state, int b, int P) = 0;
    // active B슬롯 1 decode 스텝 전진(plain=+1). h_toks/h_pos=host[B](backend가 H2D). 반환: 스텝 wall(초).
    virtual double decode_step(void* state, const int* h_toks, const int* h_pos, int B) = 0;
};

#endif // DGX_SERVING_BACKEND_H
