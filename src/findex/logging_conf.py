import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    """
    Налаштовує логування у stderr залежно від кількості прапорців -v:
    0 (-): WARNING
    1 (-v): INFO
    2 (-vv): DEBUG
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
