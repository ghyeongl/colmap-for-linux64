# logger_config.py
import logging
import sys

class LoggerFactory:
    """
    공통 로거를 생성/재활용하는 팩토리 클래스.
    - 모든 클래스에서 logger_config.LoggerFactory.get_logger(...)를 호출해
      동일한 로거 설정을 사용할 수 있도록 한다.
    """

    @classmethod
    def get_logger(cls, name=None):
        """
        지정된 name으로 로거를 반환.
        (처음 호출 시 로거에 대한 공통 설정을 수행하고, 이후에는 재활용)
        """
        if name is None:
            name = "default_logger"

        logger = logging.getLogger(name)

        if not logger.handlers:
            # 공통 로거 설정(처음 한 번만)
            logger.setLevel(logging.DEBUG)

            # 스트림 핸들러 (stdout)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)

            # 포맷 지정
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            # 2) 파일 핸들러 (DEBUG 전용) => task-debug.log
            fh_debug = logging.FileHandler("logs/task-debug.log", mode='a', encoding='utf-8')
            fh_debug.setLevel(logging.DEBUG)  # DEBUG 이상 (DEBUG, INFO, WARNING, ERROR, CRITICAL) 모두
            fh_debug.setFormatter(formatter)
            logger.addHandler(fh_debug)

            # 3) 파일 핸들러 (INFO 전용) => task-info.log
            fh_info = logging.FileHandler("logs/task-info.log", mode='a', encoding='utf-8')
            fh_info.setLevel(logging.INFO)   # INFO 이상 (INFO, WARNING, ERROR, CRITICAL)
            fh_info.setFormatter(formatter)
            logger.addHandler(fh_info)

        return logger
