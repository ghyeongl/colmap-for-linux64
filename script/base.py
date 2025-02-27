# base.py
from abc import ABC, abstractmethod

###############################################################################
# CommandRunnerBase
###############################################################################
class CommandRunnerBase(ABC):
    @abstractmethod
    def run_command(self, cmd_args, gpu_index=0):
        """
        COLMAP 등 외부 명령을 실행 (stderr를 stdout으로 합쳐 파싱).
        """
        pass


###############################################################################
# FileCheckerBase
###############################################################################
class FileCheckerBase(ABC):
    @abstractmethod
    def safe_makedirs(self, path: str):
        """
        폴더가 없으면 생성, 있으면 스킵
        """
        pass

    @abstractmethod
    def copy_tree(self, src: str, dst: str):
        """
        폴더 트리 전체를 복사
        """
        pass

    @abstractmethod
    def find_colmap_model_folders(self, base_path: str) -> list:
        """
        base_path 이하에서 cameras.bin|txt, images.bin|txt, points3D.bin|txt를 포함하는 
        'COLMAP 모델 폴더'를 찾아 리스트로 반환.
        """
        pass


###############################################################################
# ColmapPipelineBase
###############################################################################
class ColmapPipelineBase(ABC):
    @abstractmethod
    def prepare(self, input_path, output_foldername):
        pass

    @abstractmethod
    def feature_extraction(self, output_foldername, gpu_index=0):
        pass

    @abstractmethod
    def match_features(self, output_foldername, gpu_index=0):
        pass

    @abstractmethod
    def mapper(self, output_foldername, gpu_index=0):
        pass

    @abstractmethod
    def dense_reconstruction(self, output_foldername, gpu_index=0):
        pass

    @abstractmethod
    def convert_sparse_to_ply(self, output_foldername, gpu_index=0):
        """
        스파스 결과를 PLY로 변환 (파일 탐색 로직은 FileChecker 이용).
        """
        pass

    @abstractmethod
    def all_pipeline(self, input_path, output_foldername, gpu_index=0, convert_to_ply=True):
        pass
