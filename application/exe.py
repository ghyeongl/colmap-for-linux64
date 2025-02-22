#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import logging
import sys
import shutil  # Add this import at the top with other imports
import re

###############################################################################
# 로깅 설정
###############################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 스트림 핸들러(콘솔/stdout) 세팅
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
# 로그 포맷 지정
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# (선택) 파일 핸들러도 추가해 로그를 파일로 남기고 싶다면 활성화
file_handler = logging.FileHandler("task.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

###############################################################################
# 설정부
###############################################################################

# colmap 실행 경로(현재 디렉토리에 있는 심볼릭 링크가 '../colmap'을 가리킨다고 가정)
COLMAP_EXE = os.path.abspath("colmap")

###############################################################################
# 유틸 함수
###############################################################################

def run_colmap(cmd_args, gpu_index=0):
    """
    COLMAP을 실행하는 헬퍼 함수.
    - cmd_args: ['feature_extractor', '--database_path', ...] 식으로 전달
    - gpu_index: 사용하고자 하는 GPU 인덱스 (-1이면 CPU 사용)
    """
    env = os.environ.copy()
    
    if gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        logger.debug(f"Setting CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        logger.debug("GPU disabled => CUDA_VISIBLE_DEVICES='' (CPU mode)")
    
    logger.debug(f"Preparing to run colmap with cmd_args: {cmd_args}")
    
    if not os.path.exists(COLMAP_EXE):
        logger.error(f"COLMAP executable not found: {COLMAP_EXE}")
        raise FileNotFoundError(f"COLMAP executable not found: {COLMAP_EXE}")
    
    full_cmd = [COLMAP_EXE] + cmd_args
    logger.info(f"Running: {' '.join(full_cmd)}")

    try:
        result = subprocess.run(full_cmd,
                              check=True,
                              env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)

        if result.stdout:
            logger.info("[COLMAP-STDOUT]\n" + result.stdout)
            
        # Process stderr by lines
        stderr_lines = result.stderr.splitlines()
        for line in stderr_lines:
            line_stripped = line.strip()
            if not line_stripped:
                # Skip empty lines or log as debug
                logger.debug("[COLMAP-EMPTY-LINE]")
                continue
            
            if re.match(r"^F\d{8} ", line_stripped):
                # 진짜 Fatal: F20250222 형식
                logger.critical("[COLMAP] " + line_stripped)
            elif re.match(r"^E\d{8} ", line_stripped):
                # 진짜 Error
                logger.error("[COLMAP] " + line_stripped)
            elif re.match(r"^W\d{8} ", line_stripped):
                # Warning
                logger.warning("[COLMAP] " + line_stripped)
            elif re.match(r"^I\d{8} ", line_stripped):
                # Info
                logger.info("[COLMAP] " + line_stripped)
            else:
                # 나머지는 일반 메시지
                logger.debug("[COLMAP] " + line_stripped)


        logger.debug("COLMAP command finished without errors.")
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP command failed with exit code {e.returncode}")
        if e.stdout:
            logger.error("[COLMAP-STDOUT]\n" + e.stdout)
        if e.stderr:
            logger.error("[COLMAP-STDERR]\n" + e.stderr)
        raise

def safe_makedirs(path):
    """os.makedirs를 안전하게 호출"""
    logger.debug(f"Checking if directory exists: {path}")
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")
    else:
        logger.debug(f"Directory already exists: {path}")

###############################################################################
# 단계별 함수들
###############################################################################

def prepare(input_path, output_foldername):
    """
    1) 출력 폴더 생성
    2) 입력 폴더의 이미지들을 outputs/<folder>/images 폴더에 물리적으로 복사
    """
    logger.debug(f"prepare() called with input_path={input_path}, output_foldername={output_foldername}")
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_foldername)
    
    logger.debug(f"Absolute input path: {input_abs}")
    logger.debug(f"Absolute output path: {output_abs}")
    
    safe_makedirs(output_abs)
    
    images_folder = os.path.join(output_abs, "images")
    logger.debug(f"Target images folder: {images_folder}")
    
    if os.path.exists(images_folder):
        logger.debug(f"Images folder already exists: {images_folder}. Skipping copy.")
    else:
        logger.info(f"Copying images from {input_abs} to {images_folder}")
        shutil.copytree(input_abs, images_folder)
        logger.info(f"Copied images to {images_folder}")

def feature_extraction(output_foldername, gpu_index=0):
    """
    SIFT 특징점 추출
    - 데이터베이스 파일: outputs/<folder>/database.db
    - 이미지 경로: outputs/<folder>/images
    - gpu_index: GPU 인덱스 (-1: CPU 사용)
    """
    logger.debug(f"feature_extraction() called with output_foldername={output_foldername}, gpu_index={gpu_index}")
    output_abs = os.path.abspath(output_foldername)
    db_path = os.path.join(output_abs, "database.db")
    images_path = os.path.join(output_abs, "images")
    
    cmd = [
        "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_path,
        "--SiftExtraction.use_gpu", "1" if gpu_index >= 0 else "0"
    ]
    
    run_colmap(cmd, gpu_index=gpu_index)

def match_features(output_foldername, gpu_index=0):
    """
    특징점 매칭 (exhaustive_matcher)
    - 데이터베이스 파일: outputs/<folder>/database.db
    - gpu_index: GPU 인덱스 (-1: CPU 사용)
    """
    logger.debug(f"match_features() called with output_foldername={output_foldername}, gpu_index={gpu_index}")
    output_abs = os.path.abspath(output_foldername)
    db_path = os.path.join(output_abs, "database.db")
    
    cmd = [
        "exhaustive_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "1" if gpu_index >= 0 else "0"
    ]
    
    run_colmap(cmd, gpu_index=gpu_index)

def mapper(output_foldername, gpu_index=0):
    """
    Sparse Reconstruction
    - outputs/<folder>/database.db, images -> outputs/<folder>/sparse 폴더
    """
    logger.debug(f"mapper() called with output_foldername={output_foldername}")
    output_abs = os.path.abspath(output_foldername)
    db_path = os.path.join(output_abs, "database.db")
    images_path = os.path.join(output_abs, "images")
    sparse_folder = os.path.join(output_abs, "sparse")
    
    logger.debug(f"Output for sparse reconstruction: {sparse_folder}")
    safe_makedirs(sparse_folder)
    
    cmd = [
        "mapper",
        "--database_path", db_path,
        "--image_path", images_path,
        "--output_path", sparse_folder
    ]
    
    logger.debug(f"Mapper command arguments: {cmd}")
    run_colmap(cmd, gpu_index=gpu_index)

def dense_reconstruction(output_foldername, gpu_index=0):
    """
    Dense Reconstruction (image_undistorter -> patch_match_stereo -> stereo_fusion)
    - input: sparse/0 폴더
    - output: dense 폴더
    - gpu_index: GPU 인덱스 (-1: CPU 사용)
    """
    logger.debug(f"dense_reconstruction() called with output_foldername={output_foldername}, gpu_index={gpu_index}")
    output_abs = os.path.abspath(output_foldername)
    sparse_folder = os.path.join(output_abs, "sparse", "0")  # mapper 기본 출력 경로
    dense_folder = os.path.join(output_abs, "dense")
    images_path = os.path.join(output_abs, "images")
    
    logger.debug(f"Sparse folder for dense: {sparse_folder}")
    logger.debug(f"Output dense folder: {dense_folder}")
    safe_makedirs(dense_folder)
    
    # 1) image_undistorter
    cmd_undist = [
        "image_undistorter",
        "--image_path", images_path,
        "--input_path", sparse_folder,
        "--output_path", dense_folder,
        "--output_type", "COLMAP"
    ]
    logger.debug(f"image_undistorter command: {cmd_undist}")
    run_colmap(cmd_undist, gpu_index=gpu_index)
    
    # 2) patch_match_stereo
    cmd_patch = [
        "patch_match_stereo",
        "--workspace_path", dense_folder,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0" if gpu_index >= 0 else "-1"
    ]
    
    logger.debug(f"Running patch_match_stereo with GPU index {gpu_index}")
    run_colmap(cmd_patch, gpu_index=gpu_index)
    
    # 3) stereo_fusion
    fused_ply = os.path.join(dense_folder, "fused.ply")
    cmd_fusion = [
        "stereo_fusion",
        "--workspace_path", dense_folder,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply
    ]
    logger.debug(f"stereo_fusion command: {cmd_fusion}")
    run_colmap(cmd_fusion, gpu_index=gpu_index)

def all_pipeline(input_path, output_foldername, gpu_index=0):
    """
    prepare -> feature_extraction -> match_features -> mapper -> dense_reconstruction
    전체 프로세스를 한 번에 돌리는 편의 함수

    Args:
        input_path: 입력 이미지 폴더 경로
        output_foldername: 출력 폴더 경로
        gpu_index: GPU 인덱스 (기본값: 0, -1: CPU 사용)
    """
    # 1) 입력 이미지 폴더 존재 여부 확인
    if not os.path.isdir(input_path):
        logger.error(f"입력 경로가 존재하지 않습니다: {input_path}")
        return 1

    # 2) 출력 폴더 중복 확인
    if os.path.exists(output_foldername):
        logger.error(f"출력 폴더가 이미 존재합니다: {output_foldername}")
        return 1
    
    logger.info("=== Starting ALL PIPELINE ===")
    logger.debug(f"all_pipeline called with input_path={input_path}, output_foldername={output_foldername}, gpu_index={gpu_index}")
    
    prepare(input_path, output_foldername)
    feature_extraction(output_foldername, gpu_index)
    match_features(output_foldername, gpu_index)
    mapper(output_foldername, gpu_index)
    # dense_reconstruction(output_foldername, gpu_index)
    logger.info("=== All steps completed ===")


###############################################################################
# 사용 예시 (직접 실행 시)
###############################################################################
if __name__ == "__main__":
    """
    아래 코드는 예시로, 전체 파이프라인을 한 번에 실행해보고 싶다면:
    all_pipeline("inputs/datasetA", "outputs/datasetA", gpu_index=0)  # 첫 번째 GPU 사용

    실제 환경에서는 import 후
    all_pipeline(...) 등 원하는 함수를 호출하세요.
    """
    # 예) 아래 주석 해제 후 테스트:
    # all_pipeline("inputs/datasetA", "outputs/datasetA", gpu_index=1)  # 두 번째 GPU 사용
    pass
