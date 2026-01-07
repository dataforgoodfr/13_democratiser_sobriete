import logging

# --- Logging Configuration ----
LOG_LEVEL = logging.INFO
# When you set the level, all messages from a higher level of severity are also
# logged. For example, when you set the log level to `INFO`, all `WARNING`,
# `ERROR` and `CRITICAL` messages are also logged, but `DEBUG` messages are not.
# Set a seed to enable reproducibility
SEED = 1
# Set a format to the logs.
LOG_FORMAT = "[%(levelname)s | %(name)s | %(asctime)s] - %(message)s"
# Name of the file to store the logs.
LOG_FILENAME = "script_execution.log"


def configure_logging():
    """Configure logging for all modules in the project."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        force=True,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"), logging.StreamHandler()],
    )

    # Return the logger for the calling module
    return logging.getLogger()


# Configure logging when this module is imported
configure_logging()
