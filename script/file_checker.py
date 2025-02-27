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

    def find_images_recursive(self, src: str):
        """
        src 디렉토리를 재귀적으로 탐색해,
        (abs_file_path, subfolder_rel_path, filename)을 yield한다.

        예:
        src/foo/bar/001.jpg -> 
            abs_file_path:  /.../src/foo/bar/001.jpg
            subfolder_rel_path: "foo/bar"
            filename: "001.jpg"

        src/001.jpg ->
            abs_file_path:  /.../src/001.jpg
            subfolder_rel_path: ""
            filename: "001.jpg"
        """
        src_abs = os.path.abspath(src)
        for root, dirs, files in os.walk(src_abs):
            subpath = os.path.relpath(root, src_abs)
            if subpath == ".":
                subpath = ""  # 최상위 폴더 => 빈 문자열
            for f in files:
                abs_file_path = os.path.join(root, f)
                # 1) 심링크인 경우 스킵
                if os.path.islink(abs_file_path) or os.path.isdir(abs_file_path):
                    continue
                yield (abs_file_path, subpath, f)

    def copy_single_file(self, src_file, dst_file):
        """
        개별 파일을 복사 (덮어쓰기 or 스킵 여부는 필요에 따라 결정)
        """
        parent_dir = os.path.dirname(dst_file)
        if not os.path.exists(parent_dir):
            self.logger.warning(f"Destination directory does not exist, creating it: {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)
            
        if os.path.exists(dst_file):
            self.logger.warning(f"File already exists, skipping: {dst_file}")
            return
        self.logger.debug(f"Copying file {src_file} -> {dst_file}")
        shutil.copy2(src_file, dst_file)

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
