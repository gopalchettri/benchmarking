# Helper functions
# - Logging setup, file I/O, timestamp generation
# - Functions: setup_logger(), save_json(), generate_uuid()

# src/core/utils.py

import logging
import json
import uuid
from datetime import datetime, timezone

def setup_logger(log_file="results/logs/benchmark.log", level=logging.INFO):
    """
    Sets up and returns a project-wide logger.
    """
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    return logging.getLogger("llm_benchmark")

def save_json(data, file_path):
    """
    Save data as indented JSON to a file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path):
    """
    Load and return data from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_uuid():
    """
    Generate a random UUID string.
    """
    return str(uuid.uuid4())


def generate_timestamp(fmt="iso"):
    """
    Generate a current UTC timestamp.
    Formats: "iso" for ISO 8601 (with Z), "compact" for filename-safe.
    """
    now = datetime.now(timezone.utc)
    if fmt == "iso":
        return now.isoformat(timespec="seconds").replace("+00:00", "Z")
    elif fmt == "compact":
        return now.strftime("%Y%m%d_%H%M%S")
    return str(now)


def log_exception(logger, err, message=""):
    """
    Log exceptions elegantly with optional extra context message.
    """
    logger.error(f"{message} Exception: {err}", exc_info=True)
