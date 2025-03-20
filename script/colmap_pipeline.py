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
        file_checker: FileCheckerBase = None,
        gpu_index: int = 0
    ):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.gpu_index = gpu_index
        
        if command_runner is None:
            runner = CommandRunner(colmap_exe)
            self.command_runner = runner
        else:
            self.command_runner = command_runner

        self.file_checker = file_checker or FileChecker()

    def prepare(self, input_path, output_foldername):
        self.logger.info("=== Starting prepare ===")
        input_abs = os.path.abspath(input_path)
        output_abs = os.path.abspath(output_foldername)

        # 출력 폴더 생성
        self.file_checker.safe_makedirs(output_abs)

        # 이미지 폴더로 복사
        images_folder = os.path.join(output_abs, "images")
        self.file_checker.safe_makedirs(images_folder)
        for filepath, foldername, filename in self.file_checker.find_images_recursive(input_abs):
            # folder_rel = "front/bar" 등
            prefix = foldername.replace(os.path.sep, "_") if foldername else ""
            if prefix:
                dst_filename = f"{prefix}-{filename}"
            else:
                dst_filename = filename

            dst_file = os.path.join(images_folder, dst_filename)
            self.file_checker.copy_single_file(filepath, dst_file)
        
        self.logger.info("Prepare completed: all images copied into 'images/' folder.")

    def feature_extraction(self, output_foldername):
        self.logger.info("=== Starting feature_extraction ===")
        db_path = os.path.join(output_foldername, "database.db")
        images_path = os.path.join(output_foldername, "images")

        cmd = [
            "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_path,
            "--SiftExtraction.use_gpu", "1" if self.gpu_index >= 0 else "0"
        ]
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)
        self.logger.info("feature_extraction completed.")

    def match_features(self, output_foldername):
        self.logger.info("=== Starting match_features ===")
        db_path = os.path.join(output_foldername, "database.db")
        cmd = [
            "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.use_gpu", "1" if self.gpu_index >= 0 else "0"
        ]
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)
        self.logger.info("match_features completed.")

    def mapper(self, output_foldername):
        self.logger.info("=== Starting mapper (Sparse Reconstruction) ===")
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
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)
        self.logger.info("mapper (Sparse Reconstruction) completed.")

    def dense_reconstruction(self, output_foldername):
        self.logger.info("=== Starting dense_reconstruction ===")
        """
        Dense Reconstruction (image_undistorter -> patch_match_stereo -> stereo_fusion)
        """
        sparse_folder0 = os.path.join(output_foldername, "sparse", "0")
        dense_folder = os.path.join(output_foldername, "dense")
        images_folder = os.path.join(output_foldername, "images")

        self.file_checker.safe_makedirs(dense_folder)

        # 1) image_undistorter
        self._run_image_undistorter(images_folder, sparse_folder0, dense_folder)

        # 2) patch_match_stereo
        self._run_patch_match_stereo(dense_folder)

        # 3) stereo_fusion
        self._run_stereo_fusion(dense_folder)
        self.logger.info("dense_reconstruction completed.")

    def _run_image_undistorter(self, images_folder, sparse_folder, dense_folder):
        self.logger.info("Running image_undistorter...")
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
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)
        self.logger.info("image_undistorter completed.")

    def _run_patch_match_stereo(self, dense_folder):
        self.logger.info("Running patch_match_stereo...")
        """
        patch_match_stereo 실행
        """
        cmd = [
            "patch_match_stereo",
            "--workspace_path", dense_folder,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", "0" if self.gpu_index >= 0 else "-1"
        ]
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)
        self.logger.info("patch_match_stereo completed.")

    def _run_stereo_fusion(self, dense_folder):
        self.logger.info("Running stereo_fusion...")
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
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)
        self.logger.info(f"stereo_fusion completed. Output: {fused_ply}")
        
    def undistort_images(
        self,
        output_foldername,
        undistort_output_folder="undistorted"
    ):
        """
        별도의 언디스토션(Undistortion) 단계를 수행하는 메서드.
        
        일반적으로 'dense_reconstruction'에서는 내부적으로 image_undistorter가
        실행되지만, Gaussian Splatting이나 다른 NeRF 파이프라인에서
        PINHOLE/SIMPLE_PINHOLE 모델의 undistorted 이미지와 .txt 포맷의 카메라/이미지
        정보가 필요한 경우가 많아, 이 메서드를 따로 호출할 수 있음.

        Args:
            output_foldername (str): 파이프라인 출력 폴더(=mapper 후 결과가 있는 폴더).
            undistort_output_folder (str): 언디스토션 결과를 저장할 하위 폴더명.
            output_type (str): 'COLMAP' 또는 'TXT' 지정 가능.
                - COLMAP: dense_reconstruction 시 쓰는 구조(스테레오 매칭용).
                - TXT: cameras.txt, images.txt 등 텍스트 파일로 추출. (NeRF 등에서 활용)
        """
        self.logger.info("=== Starting explicit undistortion ===")
        # Sparse 결과 중 0번 폴더(기본)
        sparse_folder0 = os.path.join(output_foldername, "sparse", "0")
        images_folder = os.path.join(output_foldername, "images")

        if not os.path.exists(sparse_folder0):
            self.logger.error(f"Sparse folder not found: {sparse_folder0}")
            return 1

        undistort_folder = os.path.join(output_foldername, undistort_output_folder)
        self.file_checker.safe_makedirs(undistort_folder)

        self.logger.info(f"Undistorting images into: {undistort_folder}")
        cmd = [
            "image_undistorter",
            "--image_path", images_folder,
            "--input_path", sparse_folder0,
            "--output_path", undistort_folder
        ]
        self.command_runner.run_command(cmd, gpu_index=self.gpu_index)

        self.logger.info("=== Undistortion step completed ===")
        return 0
            
    def convert_sparse_to_ply(self, output_foldername, ply_output_dir=None):
        self.logger.info("=== Starting convert_sparse_to_ply ===")
        """
        sparse 폴더 아래의 모든 'COLMAP 모델 폴더'를 찾아 model_converter로 변환.

        Args:
            output_foldername (str): 전체 COLMAP 결과가 들어있는 상위 폴더
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
            self.command_runner.run_command(cmd, gpu_index=self.gpu_index)

        self.logger.info("Sparse-to-PLY conversion completed.")

    def all_pipeline(self, input_path, output_foldername, convert_to_ply=True):
        if not os.path.isdir(input_path):
            self.logger.error(f"Input path does not exist: {input_path}")
            return 1
        if os.path.exists(output_foldername):
            self.logger.error(f"Output folder already exists: {output_foldername}")
            return 1

        self.logger.info("=== Starting ALL PIPELINE ===")
        self.logger.debug(f"all_pipeline: input={input_path}, output={output_foldername}, gpu={self.gpu_index}")

        # 1) 준비
        self.prepare(input_path, output_foldername)
        # 2) 특징 추출
        self.feature_extraction(output_foldername)
        # 3) 특징 매칭
        self.match_features(output_foldername)
        # 4) 스파스 재구성
        self.mapper(output_foldername)

        # (옵션) PLY 변환
        if convert_to_ply:
            self.convert_sparse_to_ply(output_foldername)

        # 5) 덴스 재구성
        self.dense_reconstruction(output_foldername)

        self.logger.info("=== All steps completed ===")
        
    def all_except_dense(self, input_path, output_foldername, convert_to_ply=True):
        if not os.path.isdir(input_path):
            self.logger.error(f"Input path does not exist: {input_path}")
            return 1
        if os.path.exists(output_foldername):
            self.logger.error(f"Output folder already exists: {output_foldername}")
            return 1

        self.logger.info("=== Starting ALL EXCEPT DENSE ===")
        self.logger.debug(f"all_except_dense: input={input_path}, output={output_foldername}, gpu={self.gpu_index}")

        # 1) 준비
        self.prepare(input_path, output_foldername)
        # 2) 특징 추출
        self.feature_extraction(output_foldername)
        # 3) 특징 매칭
        self.match_features(output_foldername)
        # 4) 스파스 재구성
        self.mapper(output_foldername)

        # (옵션) PLY 변환
        if convert_to_ply:
            self.convert_sparse_to_ply(output_foldername)

        self.logger.info("=== All steps (except dense) completed ===")

