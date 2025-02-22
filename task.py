#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task.py

'from task import ...' 식으로 함수들을 불러와
특정 데이터셋에 대해 전체 COLMAP 파이프라인을 실행하는 예시.
"""

from application.exe import all_pipeline, prepare, feature_extraction, match_features, mapper, dense_reconstruction

# 샘플 입력/출력 경로 지정
input_path = "inputs/desk-images"     # 이미지를 저장한 폴더
output_path = "outputs/3dgs-images"   # COLMAP 결과 저장 폴더
USE_GPU = True
GPU_INDEX = 1 if USE_GPU else -1

# (1) 전체 파이프라인 한 번에 돌리기
# all_pipeline(input_path, output_path, GPU_INDEX)

# (2) 만약 단계별로 나누어 호출하고 싶다면 (주석 해제 후 사용)
# prepare(input_path, output_path)
# feature_extraction(output_path, GPU_INDEX)
# match_features(output_path, GPU_INDEX)
# mapper(output_path)
# dense_reconstruction(output_path, GPU_INDEX)

# ---------- (여기부터 작성) ----------
