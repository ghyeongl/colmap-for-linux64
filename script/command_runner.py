# command_runner.py
import os
import subprocess
import logging
import re
from .base import CommandRunnerBase
from .logger_config import LoggerFactory

class CommandRunner(CommandRunnerBase):
    """
    COLMAP 외부 명령 실행 구현체.
    stderr=subprocess.STDOUT로 모든 로그를 stdout에서 받아 파싱.
    """

    def __init__(self, colmap_exe):
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        self.colmap_exe = colmap_exe

    def run_command(self, cmd_args, gpu_index=0):
        env = os.environ.copy()
        if gpu_index >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            self.logger.debug(f"Setting CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            self.logger.debug("GPU disabled => CUDA_VISIBLE_DEVICES='' (CPU mode)")

        if self.colmap_exe != "colmap" and not os.path.exists(self.colmap_exe):
            err_msg = f"COLMAP executable not found: {self.colmap_exe}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        full_cmd = [self.colmap_exe] + cmd_args
        self.logger.info(f"Running: {' '.join(full_cmd)}")

        try:
            result = subprocess.run(
                full_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            self._process_output(result.stdout)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with exit code {e.returncode}")
            if e.stdout:
                self._process_output(e.stdout)
            if e.stderr:
                self._process_output(e.stderr)
            raise

    def _process_output(self, output: str):
        if not output:
            return
        for line in output.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                self.logger.debug("[EMPTY-LINE]")
                continue

            # COLMAP 로그 포맷 예: F20250222, E20250222, W20250222, I20250222
            if re.match(r"^F\d{8} ", line_stripped):
                self.logger.critical("[COLMAP] " + line_stripped)
            elif re.match(r"^E\d{8} ", line_stripped):
                self.logger.error("[COLMAP] " + line_stripped)
            elif re.match(r"^W\d{8} ", line_stripped):
                self.logger.warning("[COLMAP] " + line_stripped)
            elif re.match(r"^I\d{8} ", line_stripped):
                self.logger.debug("[COLMAP] " + line_stripped)
            else:
                self.logger.debug("[COLMAP] " + line_stripped)
