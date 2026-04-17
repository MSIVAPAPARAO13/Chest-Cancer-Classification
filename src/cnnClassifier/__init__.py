import os
import sys
import logging

# 🔥 Clear previous handlers (IMPORTANT)
logging.getLogger().handlers.clear()

# Log format
LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"

# Log directory and file
LOG_DIR = "logs"
LOG_FILE = "running_logs.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Create logs directory
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),   # ✅ fix encoding
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger("cnnClassifierLogger")
logger.setLevel(logging.INFO)

# Test log
logger.info("Logging is configured successfully")