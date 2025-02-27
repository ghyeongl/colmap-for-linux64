# colmap_pipeline.py
import os
import logging

from .base import ColmapPipelineBase, CommandRunnerBase, FileCheckerBase
from .command_runner import CommandRunner
from .file_checker import FileChecker
from .logger_config import LoggerFactory

class ColmapPipeline(ColmapPipelineBase):
    """
    COLMAP 파이프라인 구현체.
    - 폴더 탐색 로직은 FileChecker(find_colmap_model_folders)에 위임.
    - COLMAP 명령 실행은 CommandRunner(run_command)에 위임.
    """

    def __init__(
        self,
        colmap_exe="./colmap",
        command_runner: CommandRunnerBase = None,
        file_checker: FileCheckerBase = None
    ):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        
        if command_runner is None:
            runner = CommandRunner(colmap_exe)
            self.command_runner = runner
        else:
            self.command_runner = command_runner

        self.file_checker = file_checker or FileChecker()

    def prepare(self, input_path, output_foldername):
        input_abs = os.path.abspath(input_path)
        output_abs = os.path.abspath(output_foldername)

        # 출력 폴더 생성
        self.file_checker.safe_makedirs(output_abs)

        # 이미지 폴더로 복사
        images_folder = os.path.join(output_abs, "images")
        self.file_checker.copy_tree(input_abs, images_folder)

    def feature_extraction(self, output_foldername, gpu_index=0):
        db_path = os.path.join(output_foldername, "database.db")
        images_path = os.path.join(output_foldername, "images")

        cmd = [
            "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_path,
            "--SiftExtraction.use_gpu", "1" if gpu_index >= 0 else "0"
        ]
        self.command_runner.run_command(cmd, gpu_index=gpu_index)

    def match_features(self, output_foldername, gpu_index=0):
        db_path = os.path.join(output_foldername, "database.db")
        cmd = [
            "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.use_gpu", "1" if gpu_index >= 0 else "0"
        ]
        self.command_runner.run_command(cmd, gpu_index=gpu_index)

    def mapper(self, output_foldername, gpu_index=0):
        db_path = os.path.join(output_foldername, "database.db")
        images_path = os.path.join(output_foldername, "images")
        sparse_folder = os.path.join(output_foldername, "sparse")
        self.file_checker.safe_makedirs(sparse_folder)

        cmd = [
            "mapper",
            "--database_path", db_path,
            "--image_path", images_path,
            "--output_path", sparse_folder,
            "--Mapper.multiple_models", "false"
        ]
        self.command_runner.run_command(cmd, gpu_index=gpu_index)

    def dense_reconstruction(self, output_foldername, gpu_index=0):
        """
        Dense Reconstruction (image_undistorter -> patch_match_stereo -> stereo_fusion)
        """
        sparse_folder0 = os.path.join(output_foldername, "sparse", "0")
        dense_folder = os.path.join(output_foldername, "dense")
        images_folder = os.path.join(output_foldername, "images")

        self.file_checker.safe_makedirs(dense_folder)

        # 1) image_undistorter
        self._run_image_undistorter(images_folder, sparse_folder0, dense_folder, gpu_index)

        # 2) patch_match_stereo
        self._run_patch_match_stereo(dense_folder, gpu_index)

        # 3) stereo_fusion
        self._run_stereo_fusion(dense_folder, gpu_index)

    def _run_image_undistorter(self, images_folder, sparse_folder, dense_folder, gpu_index):
        """
        image_undistorter 실행
        """
        cmd = [
            "image_undistorter",
            "--image_path", images_folder,
            "--input_path", sparse_folder,
            "--output_path", dense_folder,
            "--output_type", "COLMAP"
        ]
        self.command_runner.run_command(cmd, gpu_index=gpu_index)

    def _run_patch_match_stereo(self, dense_folder, gpu_index):
        """
        patch_match_stereo 실행
        """
        cmd = [
            "patch_match_stereo",
            "--workspace_path", dense_folder,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", "0" if gpu_index >= 0 else "-1"
        ]
        self.command_runner.run_command(cmd, gpu_index=gpu_index)

    def _run_stereo_fusion(self, dense_folder, gpu_index):
        """
        stereo_fusion 실행
        """
        fused_ply = os.path.join(dense_folder, "fused.ply")
        cmd = [
            "stereo_fusion",
            "--workspace_path", dense_folder,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", fused_ply
        ]
        self.command_runner.run_command(cmd, gpu_index=gpu_index)
            
    def convert_sparse_to_ply(self, output_foldername, gpu_index=0, ply_output_dir=None):
        """
        sparse 폴더 아래의 모든 'COLMAP 모델 폴더'를 찾아 model_converter로 변환.

        Args:
            output_foldername (str): 전체 COLMAP 결과가 들어있는 상위 폴더
            gpu_index (int, optional): GPU 인덱스. 기본값 0, CPU는 -1
            ply_output_dir (str, optional): ply 파일을 모아서 저장할 폴더.
                - None이면, 각 model_folder 내부에 "points3D.ply"로 저장.
                - 지정되면, "ply_output_dir/datasetName-subfolderName.ply"로 저장.
        """
        sparse_path = os.path.join(output_foldername, "sparse")
        model_folders = self.file_checker.find_colmap_model_folders(sparse_path)

        if not model_folders:
            self.logger.info("No COLMAP model folders found under 'sparse/'. Skipping PLY conversion.")
            return

        dataset_name = os.path.basename(os.path.abspath(output_foldername))

        # (1) 만약 ply_output_dir를 지정했다면, 출력 폴더 생성
        if ply_output_dir:
            self.file_checker.safe_makedirs(ply_output_dir)
            self.logger.debug(f"PLY files will be collected in: {ply_output_dir}")

        for model_folder in model_folders:
            subfolder_name = os.path.basename(model_folder)

            if ply_output_dir:
                # (A) ply_output_dir로 모아서 저장하는 경우: datasetName-subfolderName.ply
                ply_filename = f"{dataset_name}-{subfolder_name}.ply"
                ply_path = os.path.join(ply_output_dir, ply_filename)
            else:
                # (B) model_folder 내부에 "points3D.ply"라는 이름으로 저장
                ply_path = os.path.join(model_folder, "points3D.ply")

            if os.path.exists(ply_path):
                self.logger.debug(f"PLY already exists: {ply_path}, skipping.")
                continue

            self.logger.info(f"Converting model folder {model_folder} => {ply_path}")
            cmd = [
                "model_converter",
                "--input_path", model_folder,
                "--output_path", ply_path,
                "--output_type", "PLY"
            ]
            self.command_runner.run_command(cmd, gpu_index=gpu_index)

        self.logger.info("Sparse-to-PLY conversion completed.")

    def all_pipeline(self, input_path, output_foldername, gpu_index=0, convert_to_ply=True):
        if not os.path.isdir(input_path):
            self.logger.error(f"Input path does not exist: {input_path}")
            return 1
        if os.path.exists(output_foldername):
            self.logger.error(f"Output folder already exists: {output_foldername}")
            return 1

        self.logger.info("=== Starting ALL PIPELINE ===")
        self.logger.debug(f"all_pipeline: input={input_path}, output={output_foldername}, gpu={gpu_index}")

        # 1) 준비
        self.prepare(input_path, output_foldername)
        # 2) 특징 추출
        self.feature_extraction(output_foldername, gpu_index)
        # 3) 특징 매칭
        self.match_features(output_foldername, gpu_index)
        # 4) 스파스 재구성
        self.mapper(output_foldername, gpu_index)

        # (옵션) PLY 변환
        if convert_to_ply:
            self.convert_sparse_to_ply(output_foldername, gpu_index)

        # 5) 덴스 재구성
        self.dense_reconstruction(output_foldername, gpu_index)

        self.logger.info("=== All steps completed ===")
