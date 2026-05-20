import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from src.config.settings import settings
import src.config.settings as settings_module
from src.config.logging_config import get_logger
logger = get_logger("diagnostic")


print("Settings module file:", settings_module.__file__)

if __name__ == "__main__":
    logger.info("==========================================")
    logger.info("Engineer diagnostic")
    logger.info(f"BASE_DIR : {settings.BASE_DIR}")
    logger.info(f"ENV : {settings.ENV}")
    logger.info(f"GROQ_KEY : {settings.GROQ_API_KEY}")
    logger.info("==========================================")
