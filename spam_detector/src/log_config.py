import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

log_format = (
    " | "
    "<level>{level: <8}</level> | "
    "<cyan>{file}</cyan>:<cyan>{line}</cyan>:(<cyan>{function}</cyan>) "
    "- <level>{message}</level>"
)


def setup_logging() -> None:
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/bot.log")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logs_dir = Path(log_file_path).parent
    logs_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format=log_format, level="DEBUG")
    logger.add(
        log_file_path,
        format=log_format,
        level=log_level,
        encoding="utf-8",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )
    logger.add(
        logs_dir / "training_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="DEBUG",
        encoding="utf-8",
        rotation="1 day",
        retention="30 days",
        filter=lambda record: "Training" in record["message"]
        or "Epoch" in record["message"]
        or "Model saved" in record["message"]
        or "device" in record["message"],
    )


setup_logging()
