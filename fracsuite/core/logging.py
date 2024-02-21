import logging
from rich.console import Console
from rich.logging import RichHandler

logger: logging.Logger = None

def start(name, debug = False):
    # Erstellt eine Rich Console
    console = Console()

    # Konfiguriert das Logging, um RichHandler zu verwenden
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
    )

    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if not debug else logging.DEBUG)


def info(*msg):
    logger.info(*msg)

def debug(*msg):
    logger.debug(*msg)

def warning(*msg):
    logger.warning(*msg)

def error(*msg):
    logger.error(*msg)

def critical(*msg):
    logger.critical(*msg)

def exception(*msg):
    logger.exception(*msg)
