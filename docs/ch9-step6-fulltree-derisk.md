# 챕터9 tree-spec STEP 6.0a — full multi-level forking tree accept de-risk → **NO-GO**

> 가설(STEP 6 의뢰): leaf-alt(d=1만 width)가 accept 2.86서 천장 미달 → **각 레벨 fork**면 top-4 0.949 누적 4-5.
> de-risk(per-depth rank 측정, `test_treespec_depth`): **NO-GO — full forking은 ≤8(or SB)서 4-5 못 냄. leaf-alt가 사실상 천장.**
> 큰 빌드(MTP KV forking) 착수 전 측정으로 차단 — 챕터9 de-risk-first 원칙대로.

## 핵심 측정 — MTP는 예측 깊이가 깊어질수록 degrade

기존 de-risk(test_treespec)는 **depth-1만**: top-4 0.949(즉시 다음토큰). full forking은 각 depth fork라 **per-depth**
필요. teacher-forced chain(true 토큰 먹이되 d≥2는 MTP own hidden = tree 수용-path conditioning과 동일)로 측정:

| depth | top-1 | top-2 | top-4 | (의미) |
|-------|-------|-------|-------|--------|
| d=1 | 0.787 | 0.912 | **0.975** | 즉시 다음 — 높음 |
| d=2 | 0.738 | 0.812 | 0.900 | |
| d=3 | 0.700 | 0.812 | 0.875 | |
| d=4 | 0.662 | 0.800 | 0.838 | |
| d=6 | 0.675 | 0.762 | 0.812 | 깊을수록 ↓ |

(27B W4A8, N=80, 단일 프롬프트. N=48과 동일 추세 — 견고.) **"top-4 0.949"는 d=1 수치.** 깊이서 MTP가 자기-conditioning
(own hidden)으로 degrade → top-4가 d≥2서 0.81~0.90.

## 천장 계산 — full top-w tree accept = Σ_{d=1}^D Π_{i=1}^d P(rank<w @ i)

| shape | accept | nodes | zone |
|-------|--------|-------|------|
| chain(top-1) depth-6 | 2.36 | 7 | ≤8 |
| top-2 depth-4 | 2.74 | 31 | SB(>8) |
| top-2 depth-6 | ~3.35 | 127 | 불가 |
| top-4 depth-6 | ~4.24 | **5461** | 불가 |

**accept 4-5는 수천 노드(top-4 depth-6)서만** — verify Sq 수천 = 완전 비현실(amortize·메모리 불가).
**feasible(≤32 SB)서 최대 ~2.7**(top-2 depth-4, 31노드).

## 결정타 — leaf-alt가 full forking보다 **우수**

- 측정된 leaf-alt(STEP 4, K=6 W=2): **accept 2.86, ≤8노드.**
- full top-2 depth-4(forking): accept 2.74, **31노드**(SB 확장 필요).
- → **leaf-alt가 더 높고 4× 싸다.** 이유: leaf-alt는 예산을 **depth(chain, 노드당 저렴)+d=1 width(최고가치 0.912)**에
  쓰고, full forking은 저가치 deep width(d≥2 top-2 ~0.8)에 노드 낭비.

## 결론 (NO-GO, 정직)
- **full multi-level forking tree 빌드 안 함.** MTP가 깊이서 degrade → 천장 4-5는 비현실 노드수서만. feasible서 ~2.7.
- **현 leaf-alt(STEP 4)가 사실상 천장** — chain depth(amortize 저렴) + d=1 width(최고가치)가 옳은 예산 배분.
- de-risk가 **큰 빌드(MTP KV forking, GDN tree-fork의 MTP 버전)를 차단** — 측정 0.5일 vs 빌드 수일+페이오프 없음.
- 챕터9 교훈 적용: 하드파트 de-risk 먼저 → 페이오프 없으면 안 짓는다(챕터7 cp.async/FP4 기각과 동형).

## decode-spec 좌표 (현 상태로 종료)
- tree-spec(leaf-alt, STEP 4) + MTP q4(STEP 5) = **decode 18.8 t/s, accept 2.3~2.9, mism=0.** 이게 단일-MTP-head
  speculation의 실용 천장. 추가 accept↑ lever는 **MTP head 자체 개선(더 정확한 draft, 다중 head)** — 모델/학습 영역,
  커널 최적화 아님.
- 남은 커널 lever(별개): verify Sq>8 SB확장(batch serving), prefill 등.
