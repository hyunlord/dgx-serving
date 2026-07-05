# M7 Phase 1 — 공개 직전 체크리스트 ("push만 남음" 상태)

> 결정 반영 완료. **공개(push)는 사용자 손** — 아래 전부 ✓ 확인 후 남은 것은 사용자의 push 한 번.

## 체크리스트 (measured)

| 항목 | 상태 | 근거 |
|---|---|---|
| **LICENSE** | ✅ | `LICENSE`(Apache-2.0 전문) 추가·README 배지·라이선스 절 |
| **README** | ✅ | 초안→라이브 확정. 연구 톤·NVIDIA 상표 고지·정직 벤치("진다" 좌표 유지)·문서 링크·GB10 전제·가중치 미배포 고지 |
| **경로 scrub** | ✅ | 전 문서/코드 로컬 경로·개인식별 **0건**(git grep 검증). placeholder화(의미 보존) |
| **worklog 분류** | ✅ | `m7-worklog-triage.md`(Tier1 공개/Tier2 권장/Tier3 보류 후보). 정리는 push 전 사용자 결정 |
| **빌드 스모크** | ✅ | `make all` exit 0·53 테스트 바이너리 컴파일(nvcc 13.0·GB10·sm_120)·에러 0(경고 2 무해)·router ISO 실행 PASS(idx 100%) |
| **비밀 0** | ✅ | git 이력 전수 스캔=비밀/키/토큰 0·가중치 커밋 이력 0·gitignore 견고(weights·.bin·.venv·.omc 차단) |
| **author 이메일** | ✅ | 향후 커밋 repo-local noreply(`git config --local user.email` = users.noreply). 과거 이력 재작성 안 함(비밀 0 확인됨) |

## push 전 사용자 선택(선택 — 안 해도 공개 가능)

- **worklog 정리**(선택): curated 원하면 Tier3·m7-* 메타 문서를 push 전 `git rm` 또는 `docs/worklog/` 이동
  (`m7-worklog-triage.md` 참조). **기본=그대로 공개**(README가 "개발 로그" 라벨로 기대치 관리).
- **noreply numeric-ID**(선택): 커밋을 GitHub 프로필에 연결하려면 `ID+hyunlord@users.noreply.github.com`
  형식으로 조정(GitHub Settings→Emails에서 확인). 현재 username 형식도 개인 이메일 은닉은 달성.
- **26B 파레토·spec 경계 등 알려진 갭**: README "알려진 갭"에 로드맵으로 명시됨 → 완벽 대기 불필요(즉시 공개 가능).

## 남은 것 = push (사용자 실행)

```bash
# 1) (신규 공개면) GitHub에 빈 repo 생성 후 remote 연결
git remote add origin git@github.com:<사용자>/dgx-serving.git

# 2) 현재 main 브랜치 push
git push -u origin main
```

- ★이 세션은 push를 실행하지 않는다(공개는 사용자 손). 위 상태 = **"push 한 번만 남은" 공개 준비 완료**.
- push 후 = M7 공개 = 처음 비전("장비×모델 최적 서빙 + 대역폭 한계 탐사")의 정직한 공개 완성.
