from .logger import info
from options import args
from .distributed import init_distributed

__all__ = ['info', 'init_distributed']
