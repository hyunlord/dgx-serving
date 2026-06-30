#!/usr/bin/env python3
"""기존 q4 weight 덤프의 fp32 scale → fp16 변환 (decode/verify 대역폭 -10%, scale가 트래픽의 ~20%).

챕터9: 27B 덤프가 fp32 scale(scale_dtype 미설정=0)이었음 — scale가 bytes/token의 20%. fp16이면 절반(트래픽 -10%).
이 변환은 extract_weights_qwen35.py --scale-dtype fp16 재덤프와 **정확히 동일**: quantize_int4가 만든 fp32 scale을
fp16으로 cast(저장만 fp16). rel err 9.3e-4 안전(docs/02d). gemv_w4a8/gemv_q4가 scale_dtype=1(__half) 지원 — 커널 변경 0.

src의 *_q4.bin·embed·norm 등은 dst에 symlink(12.8GB 복사 회피), *_scale.bin만 fp16 변환, manifest에 scale_dtype=1.
사용법: python tools/convert_scale_fp16.py <src_weights_dir> <dst_weights_dir>
"""
import os, sys, json, numpy as np

def main():
    if len(sys.argv) < 3:
        print("usage: convert_scale_fp16.py <src_weights_dir> <dst_weights_dir>"); sys.exit(1)
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.makedirs(dst, exist_ok=True)
    nconv = nlink = 0; old = new = 0
    for f in os.listdir(src):
        sp, dp = os.path.join(src, f), os.path.join(dst, f)
        if os.path.lexists(dp): os.remove(dp)
        if f.endswith("_scale.bin"):
            a = np.fromfile(sp, dtype=np.float32)
            a.astype(np.float16).tofile(dp)
            old += a.nbytes; new += a.nbytes // 2; nconv += 1
        elif f == "manifest.json":
            m = json.load(open(sp)); m["scale_dtype"] = 1
            json.dump(m, open(dp, "w"), indent=2)
        else:
            os.symlink(sp, dp); nlink += 1
    print(f"scale fp32->fp16: {nconv}개 ({old/1e9:.2f}GB->{new/1e9:.2f}GB, -{(old-new)/1e9:.2f}GB) + {nlink} symlink + manifest scale_dtype=1")
    print(f"  dst={dst}  (== extract --scale-dtype fp16 재덤프와 동일, rel 9.3e-4, gemv scale_dtype=1 자동)")

if __name__ == "__main__":
    main()
