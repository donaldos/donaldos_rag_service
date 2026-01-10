import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str = "app",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    공통 로거 생성 함수
    - 콘솔 + 파일 로깅
    - INFO/ERROR 분리
    - 로그 회전 지원
    """

    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # 중복 로그 방지

    if logger.handlers:
        return logger  # 중복 핸들러 방지

    # ------------------------
    # 로그 포맷
    # ------------------------
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d] %(message)s"
    )

    # ------------------------
    # 콘솔 핸들러
    # ------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # ------------------------
    # 일반 로그 파일 (INFO 이상)
    # ------------------------
    file_handler = RotatingFileHandler(
        filename=f"{log_dir}/app.log",
        maxBytes=5 * 1024 * 1024,   # 5MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ------------------------
    # 에러 로그 파일 (ERROR 이상)
    # ------------------------
    error_handler = RotatingFileHandler(
        filename=f"{log_dir}/app.error.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # ------------------------
    # 핸들러 등록
    # ------------------------
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger
