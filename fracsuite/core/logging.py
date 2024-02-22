import logging
from rich.console import Console
from rich.logging import RichHandler

logger: logging.Logger = None

def start(name, debug = False):
    global logger
    if logger is not None:
        logger.info("Logger already started, skipping.")
        logger.setLevel(logging.INFO if not debug else logging.DEBUG)

        return


    # Erstellt eine Rich Console
    console = Console()

    # Konfiguriert das Logging, um RichHandler zu verwenden
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if not debug else logging.DEBUG)


def info(*msg):
    if logger is None:
        print(*msg)
        return
    logger.info(*msg)

def debug(*msg):
    if logger is None:
        print(*msg)
        return
    logger.debug(*msg, stacklevel=2)

def warning(*msg):
    if logger is None:
        print(*msg)
        return
    logger.warning(*msg)

def error(*msg):
    if logger is None:
        print(*msg)
        return
    logger.error(*msg)

def critical(*msg):
    if logger is None:
        print(*msg)
        return
    logger.critical(*msg)

def exception(*msg):
    if logger is None:
        print(*msg)
        return
    logger.exception(*msg)
