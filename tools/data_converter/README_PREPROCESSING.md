# NavSim 맵 데이터 사전 생성 가이드

## 목적
학습 중 맵 쿼리로 인한 병목 현상(iteration당 11.6초 소요)을 제거하기 위해 맵 벡터를 미리 생성하여 PKL 파일에 저장합니다.

## 실행 방법

```bash
cd /home/byounggun/MapTR

# 16개 프로세스로 사전 생성 (534k 샘플 기준 약 2-4시간 소요)
python tools/data_converter/pregenerate_navsim_maps.py \
    --input-pkl data/navsim/navsim_map_infos_trainval_filtered.pkl \
    --output-pkl data/navsim/navsim_map_infos_trainval_with_maps.pkl \
    --data-root data/navsim \
    --nproc 16
```

## 설정 파일 업데이트

전처리 완료 후, 학습 설정 파일을 수정하세요:

```python
# projects/configs/maptr/maptr_tiny_r50_navsim_24e.py
train=dict(
    ...
    ann_file=data_root + 'navsim_map_infos_trainval_with_maps.pkl',  # <-이것으로 변경
    ...
)
```

## 기대 효과
- **현재**: iteration당 14.5초 (데이터 로딩 11.6초 + 학습 3초)
- **적용 후**: iteration당 3-4초 (데이터 로딩 0.5초 + 학습 3초)
- **속도 향상**: 약 4배

## 주의사항
- 생성된 PKL 파일 크기: 약 50-100GB (기존 ~2GB 대비 증가)
- 한 번만 실행하면 되고, 이후 학습은 훨씬 빠릅니다
- dataset 코드가 자동으로 미리 생성된 맵을 감지하여 사용합니다
- 미리 생성된 데이터가 없으면 자동으로 런타임 생성 모드로 전환됩니다
