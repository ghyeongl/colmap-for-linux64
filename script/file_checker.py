# file_checker.py
import os
import shutil
import logging
from .base import FileCheckerBase
from .logger_config import LoggerFactory

class FileChecker(FileCheckerBase):
    """
    파일/폴더 관련 유틸리티 구현체.
    - safe_makedirs
    - copy_tree
    - find_colmap_model_folders
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

    def safe_makedirs(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
            self.logger.info(f"Created directory: {path}")
        else:
            self.logger.debug(f"Directory already exists: {path}")

    def copy_tree(self, src: str, dst: str):
        if os.path.exists(dst):
            self.logger.debug(f"Already exists: {dst}, skip copy_tree.")
        else:
            self.logger.info(f"Copying from {src} to {dst}")
            shutil.copytree(src, dst)

    def find_colmap_model_folders(self, base_path: str) -> list:
        """
        base_path 이하를 os.walk로 돌며
        cameras.bin|txt, images.bin|txt, points3D.bin|txt가 존재하는 폴더를 모두 찾아 리스트로 반환.
        """
        model_folders = []
        base_path = os.path.abspath(base_path)
        if not os.path.isdir(base_path):
            self.logger.debug(f"find_colmap_model_folders: {base_path} is not a dir.")
            return model_folders

        for root, dirs, files in os.walk(base_path):
            if self._is_colmap_model_folder(root):
                model_folders.append(root)

        return model_folders

    def _is_colmap_model_folder(self, folder: str) -> bool:
        cameras_bin = os.path.join(folder, "cameras.bin")
        images_bin = os.path.join(folder, "images.bin")
        points3d_bin = os.path.join(folder, "points3D.bin")

        cameras_txt = os.path.join(folder, "cameras.txt")
        images_txt = os.path.join(folder, "images.txt")
        points3d_txt = os.path.join(folder, "points3D.txt")

        has_bin = (os.path.exists(cameras_bin) and
                   os.path.exists(images_bin) and
                   os.path.exists(points3d_bin))
        has_txt = (os.path.exists(cameras_txt) and
                   os.path.exists(images_txt) and
                   os.path.exists(points3d_txt))

        return (has_bin or has_txt)
