import sys
import logging
import colorlog

# Define log levels as constants
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

DEFAULT_LOG_LEVEL = 'INFO'

def setup_logger(
    name: str = __name__,
    level: str = DEFAULT_LOG_LEVEL,
    propagate: bool = False,
    clear_existing: bool = True
) -> logging.Logger:
    logger = colorlog.getLogger(name)
    
    if clear_existing:
        logger.handlers = []
        logger.propagate = propagate
        
        # Clear root logger handlers if they use colorlog
        root_logger = logging.getLogger()
        if any(isinstance(h, colorlog.StreamHandler) for h in root_logger.handlers):
            root_logger.handlers = []
    
    if not logger.handlers:  # Only add handlers if none exist
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setFormatter(colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'white,bg_red',
            }
        ))
        logger.addHandler(handler)
    
    logger.setLevel(LOG_LEVELS.get(level, LOG_LEVELS[DEFAULT_LOG_LEVEL]))
    return logger

def configure_root_logger(level: str = DEFAULT_LOG_LEVEL):
    logging.basicConfig(level=LOG_LEVELS.get(level, LOG_LEVELS[DEFAULT_LOG_LEVEL]))
    setup_logger('root', level)