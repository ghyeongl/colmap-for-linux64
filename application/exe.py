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
        "--output_path", sparse_folder,
        "--Mapper.multiple_models", "false"
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
    
def convert_all_sparse_to_ply(parent_folder: str, output_dir: str = None):
    """
    주어진 parent_folder 아래를 재귀적으로 뒤져서,
    - 'sparse'라는 이름의 폴더를 찾고,
    - 그 안의 서브폴더(예: sparse/0, sparse/1 등)에 cameras.bin|txt, images.bin|txt, points3D.bin|txt 세트가 있으면
      model_converter를 이용해 PLY 파일을 생성.

    Args:
        parent_folder (str): 최상위 폴더. 이 폴더 아래 모든 디렉토리를 탐색.
        output_dir (str, optional): 변환된 PLY를 저장할 상위 폴더. 지정하지 않으면 해당 sparse/<subfolder> 내부에 생성.
    """
    parent_folder = os.path.abspath(parent_folder)
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting auto-conversion under {parent_folder}")

    # os.walk를 통해 parent_folder 안의 모든 하위 디렉토리를 탐색
    for root, dirs, files in os.walk(parent_folder):
        # 'sparse'라는 폴더가 있으면 그 내부의 서브폴더들을 뒤질 수 있도록 처리
        # 하지만 os.walk는 이미 모든 디렉토리를 재귀적으로 순회하므로,
        # "root"가 "sparse"로 끝나면 그 안의 0,1,2,... 폴더를 찾으면 됨.
        # 예: root.endswith(os.path.sep + "sparse")
        #     or root.endswith("sparse")  # 경로 구분자 처리
        # 대신 root가 "sparse/어떤숫자"인 경우 바로 처리해도 됨.
        
        # 1) 혹시 "sparse"라는 이름 그 자체인 디렉토리인지 검사
        if os.path.basename(root) == "sparse":
            # sparse/ 아래에 있는 디렉토리를 찾는다 (예: 0, 1, 2, ...)
            subfolders = [os.path.join(root, d) for d in dirs]
            # subfolders 각각에 대해 모델이 있으면 변환
            for sub in subfolders:
                _try_convert_model_to_ply(sub, output_dir)
        
        # 2) 혹은 "sparse/0", "sparse/1"처럼 이미 버전 폴더 내부라면?
        #    os.walk 때문에 결국 이 루프에서 root가 sparse/0이 되는 순간이 옴.
        #    따라서 basename이 0,1,2,...인 경우에도 시도 가능.
        #    다만, basename이 0인지 1인지 모를 수도 있고, "some_name"일 수도 있으니,
        #    아래와 같이 간단히 처리할 수도 있음
        #    (선택사항: 둘 중 한 방법만 써도 충분히 동작함)
        if is_colmap_model_folder(root):
            _try_convert_model_to_ply(root, output_dir)

def is_colmap_model_folder(folder: str) -> bool:
    """
    해당 폴더 안에 cameras.bin|txt, images.bin|txt, points3D.bin|txt 세 파일이 존재하면 True
    """
    cameras_bin = os.path.join(folder, "cameras.bin")
    images_bin = os.path.join(folder, "images.bin")
    points3d_bin = os.path.join(folder, "points3D.bin")

    cameras_txt = os.path.join(folder, "cameras.txt")
    images_txt = os.path.join(folder, "images.txt")
    points3d_txt = os.path.join(folder, "points3D.txt")

    has_bin = os.path.exists(cameras_bin) and os.path.exists(images_bin) and os.path.exists(points3d_bin)
    has_txt = os.path.exists(cameras_txt) and os.path.exists(images_txt) and os.path.exists(points3d_txt)

    return has_bin or has_txt

def _try_convert_model_to_ply(model_folder: str, output_dir: str = None):
    """
    인풋 조건: 입력된 폴더 첫 번째 레벨에 cameras.bin|txt, images.bin|txt, points3D.bin|txt 파일이 있어야 함.
    model_folder 안에 COLMAP 모델 파일( .bin/.txt )이 있으면 PLY로 변환
    변환된 PLY를 output_dir가 있으면 거기에 저장, 없으면 model_folder 내부에 저장.
    이미 PLY가 있으면(또는 변환 불가능하면) 건너뜀.
    """
    if not is_colmap_model_folder(model_folder):
        return

    folder_abs = os.path.abspath(model_folder)
    folder_name = os.path.basename(folder_abs)
    sparse_folder = os.path.dirname(folder_abs)            # ~/project_name/dataset_name/sparse
    dataset_folder = os.path.dirname(sparse_folder)        # ~/project_name/dataset_name
    dataset_name = os.path.basename(dataset_folder)        # "dataset_name"
    
    # 최종 파일명: dataset_name-folder_name.ply => 예) dataset_name-0.ply
    ply_filename = f"{dataset_name}-{folder_name}.ply"
    
    # 만약 output_dir가 주어지면, 그 경로에 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ply_path = os.path.join(output_dir, ply_filename)
    else:
        # 지정하지 않으면 model_folder 내부에 저장
        # (예: ~/project_name/dataset_name/sparse/0/dataset_name-0.ply)
        ply_path = os.path.join(folder_abs, "points3D.ply")
    
    if os.path.exists(ply_path):
        logger.debug(f"PLY file already exists, skipping: {ply_path}")
        return

    logger.info(f"Converting model in {model_folder} => {ply_path}")
    cmd = [
        "model_converter",
        "--input_path", folder_abs,
        "--output_path", ply_path,
        "--output_type", "PLY"
    ]
    run_colmap(cmd)
    logger.info(f"Done converting {folder_abs} to {ply_path}")

def convert_sparse_to_ply(output_foldername):
    """
    스파스 재구성 결과를 PLY 포맷으로 변환
    - input: sparse/0 폴더
    - output: sparse/points3D.ply 파일
    """
    logger.debug(f"convert_sparse_to_ply() called with output_foldername={output_foldername}")
    output_abs = os.path.abspath(output_foldername)
    sparse_folder = os.path.join(output_abs, "sparse", "0")
    
    _try_convert_model_to_ply(sparse_folder)

def merge_sparse_models(output_foldername):
    """
    Merge multiple sparse reconstruction models in the sparse folder.
    - Looks for numeric subfolders (0,1,2..) under outputs/{folder}/sparse/
    - Merges them sequentially using colmap model_merger
    - Final result is saved in sparse/merged/
    """
    sparse_path = os.path.join(output_foldername, "sparse")
    if not os.path.isdir(sparse_path):
        logger.error(f"sparse folder not found: {sparse_path}")
        return

    # Find numeric subfolders (0,1,2...)
    subfolders = []
    for name in os.listdir(sparse_path):
        folder_path = os.path.join(sparse_path, name)
        if re.match(r"^\d+$", name) and os.path.isdir(folder_path):
            subfolders.append(name)

    subfolders.sort(key=lambda x: int(x))
    logger.info(f"Found sparse models to merge: {subfolders}")
    if len(subfolders) < 2:
        logger.info("Need at least two sparse models to merge. Skipping.")
        return

    merged_path = os.path.join(sparse_path, "merged")
    os.makedirs(merged_path, exist_ok=True)

    # Merge first two models
    model1 = os.path.join(sparse_path, subfolders[0])
    model2 = os.path.join(sparse_path, subfolders[1])
    logger.info(f"Merging initial models {subfolders[0]} and {subfolders[1]}")
    cmd = [
        "model_merger",
        "--input_path1", model1,
        "--input_path2", model2,
        "--output_path", merged_path
    ]
    run_colmap(cmd)

    # Chain-merge remaining models
    for idx in range(2, len(subfolders)):
        next_model = os.path.join(sparse_path, subfolders[idx])
        logger.info(f"Merging model {subfolders[idx]} into merged result")
        cmd = [
            "model_merger",
            "--input_path1", merged_path,
            "--input_path2", next_model,
            "--output_path", merged_path
        ]
        run_colmap(cmd)

    logger.info(f"All sparse models merged successfully to: {merged_path}")

def all_pipeline(input_path, output_foldername, gpu_index=0, convert_to_ply=True):
    """
    prepare -> feature_extraction -> match_features -> mapper -> [convert_to_ply] -> dense_reconstruction
    전체 프로세스를 한 번에 돌리는 편의 함수

    Args:
        input_path: 입력 이미지 폴더 경로
        output_foldername: 출력 폴더 경로
        gpu_index: GPU 인덱스 (기본값: 0, -1: CPU 사용)
        convert_to_ply: 스파스 결과를 PLY로 변환할지 여부 (기본값: True)
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
    dense_reconstruction(output_foldername, gpu_index)
    if convert_to_ply:
        convert_all_sparse_to_ply(output_foldername)
    
    
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
