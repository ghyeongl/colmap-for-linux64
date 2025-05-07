#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task.py

절차: Inputs의 이미지 중, Transient와 Static 이미지셋에 대해 각각 Colmap 파이프라인 진행 (Undistortion까지)
Undistorted static 이미지셋을 Transient 이미지와 바꿈
"""

import script

# 샘플 입력/출력 경로 지정
input_path = "inputs/static-bicycle-exp"     # 이미지를 저장한 폴더
output_path = "outputs/bicycle-blue-green/static-1xsize"   # COLMAP 결과 저장 폴더
input_path2 = "inputs/transient-bicycle-exp"     # 이미지를 저장한 폴더
output_path2 = "outputs/bicycle-blue-green/transient-1xsize"   # COLMAP 결과 저장 폴더
USE_GPU = True
GPU_INDEX = 1 if USE_GPU else -1
cp = script.ColmapPipeline(gpu_index=GPU_INDEX)

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

cp.prepare(input_path, output_path)
# cp.resize_images(output_path)
cp.feature_extraction(output_path)
cp.match_features(output_path)
cp.mapper(output_path)
cp.convert_sparse_to_ply(output_path)
cp.dense_reconstruction(output_path)

cp.prepare(input_path2, output_path2)
# cp.resize_images(output_path2)
cp.feature_extraction(output_path2)
cp.match_features(output_path2)
cp.mapper(output_path2)
cp.convert_sparse_to_ply(output_path2)
cp.dense_reconstruction(output_path2)
