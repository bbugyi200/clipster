"""Helper functions used by setup.py."""

from pathlib import Path
import re
from typing import Dict, Iterator, List, Optional, Union


PathLike = Union[str, Path]


def install_requires() -> List[str]:
    return list(_requires("requirements.txt"))


def extras_require() -> Dict[str, List[str]]:
    result = {}

    reqtxt = "requirements.txt"
    for extra in _collect_extras(reqtxt):
        result[extra] = list(_requires(reqtxt, extra=extra))

    return result


def _requires(reqtxt_basename: str, extra: str = None) -> Iterator[str]:
    reqtxt = Path(__file__).parent / reqtxt_basename
    for line in reqtxt.open():
        line = line.strip()

        if not line or line.lstrip().startswith(("#", "-")):
            continue

        package = line.split(" ")[0].strip()

        if extra != _get_extra(line):
            continue

        yield package


def _collect_extras(reqtxt: PathLike) -> Iterator[str]:
    reqtxt = Path(reqtxt)
    extra_set = set()
    for line in reqtxt.open():
        extra = _get_extra(line)
        if extra is None:
            continue

        if extra not in extra_set:
            yield extra
            extra_set.add(extra)


def _get_extra(line: str) -> Optional[str]:
    if not re.match(r"^[A-Za-z].* # \S+\s*$", line):
        return None

    return line.split("#")[1].strip()
