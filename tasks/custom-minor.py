#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task.py

'from task import ...' 식으로 함수들을 불러와
특정 데이터셋에 대해 전체 COLMAP 파이프라인을 실행하는 예시.
"""

import script

# 샘플 입력/출력 경로 지정
input_path = "inputs/desk-images"     # 이미지를 저장한 폴더
output_path = "outputs/3dgs-images"   # COLMAP 결과 저장 폴더
USE_GPU = True
GPU_INDEX = 4 if USE_GPU else -1
cp = script.ColmapPipeline()

# (1) 전체 파이프라인 한 번에 돌리기 (주석 해제해서 사용)
# cp.all_pipeline(input_path, output_path, gpu_index=GPU_INDEX, convert_to_ply=True)

# (2) 단계별로 나누어 호출하고 싶다면 (주석 해제 후 사용)
# cp.prepare(input_path, output_path)
# cp.feature_extraction(output_path, GPU_INDEX)
# cp.match_features(output_path, GPU_INDEX)
# cp.mapper(output_path, GPU_INDEX)
# cp.convert_sparse_to_ply(output_path, GPU_INDEX)
# cp.dense_reconstruction(output_path, GPU_INDEX)

# 필요한 경우 merge_sparse_models 같은 부가기능도 클래스 메서드로 호출할 수 있음
# cp.merge_sparse_models(output_path, GPU_INDEX)

# ---------- (여기부터 작성) ----------

# cp.all_except_dense("inputs/k-bicycle-export", "outputs/k-bicycle-export", GPU_INDEX)
cp.all_pipeline("inputs/static-bicycle-exp", "outputs/static-bicycle-exp", GPU_INDEX)
