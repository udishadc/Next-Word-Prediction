import os
import json
import logging
from datetime import datetime


def setup_logger(log_dir="logs"):
    """Console + file logger with a timestamped run ID."""
    os.makedirs(log_dir, exist_ok=True)

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{run_id}.log")

    logger = logging.getLogger("sherlock")
    logger.setLevel(logging.DEBUG)

    # console handler — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    # file handler — DEBUG and above
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Run ID   : {run_id}")
    logger.info(f"Log file : {log_file}")

    return logger, run_id


def save_run_summary(history, metrics, config, run_id, log_dir="logs"):
    """Persist config, metrics, and training history to a JSON file."""
    summary = {
        "run_id"   : run_id,
        "timestamp": datetime.now().isoformat(),
        "config"   : config,
        "metrics"  : metrics,
        "history"  : {
            "train_loss"  : history["train_loss"],
            "val_loss"    : history["val_loss"],
            "train_acc"   : history["train_acc"],
            "val_acc"     : history["val_acc"],
            "val_top5_acc": history["val_top5_acc"],
        }
    }

    summary_path = os.path.join(log_dir, f"summary_{run_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path