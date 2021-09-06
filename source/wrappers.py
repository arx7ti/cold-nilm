#!/usr/bin/env python3

from beartype import beartype
from typing import Callable, Dict, Tuple, Optional
from pathos.multiprocessing import Pool
from configparser import ConfigParser


class CustomConfigParser(ConfigParser):

    path: Optional[str]

    @beartype
    def __init__(self, path: Optional[str] = None) -> None:
        super(CustomConfigParser, self).__init__()
        self.path = path
        del path
        if self.path is not None:
            self.read(self.path)
        return None


@beartype
def wrap_generator(
    generator: Callable, tasks: Dict[Tuple[str, ...], int], n_jobs: int
) -> None:
    """
    Arguments:
        generator: Callable
        tasks: Dict[Tuple[str, ...], int]
        n_jobs: int
    Returns:
        None
    """
    pool = Pool(n_jobs)
    pool.starmap(generator, enumerate(tasks.items()))
    pool.close()
    pool.join()
    return None
