"""Helper functions used by setup.py."""

from pathlib import Path
import re
from typing import Dict, Iterator, List, Union


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

        package = line.split("#")[0].strip()
        line_extras = _get_extras(line)

        if extra is None and line_extras:
            continue

        if extra is not None and extra not in line_extras:
            continue

        yield package


def _collect_extras(reqtxt: PathLike) -> Iterator[str]:
    reqtxt = Path(reqtxt)
    extra_set = set()
    for line in reqtxt.open():
        extras = _get_extras(line)
        for extra in extras:
            if extra not in extra_set:
                yield extra
                extra_set.add(extra)


def _get_extras(line: str) -> List[str]:
    if not re.match(r"^[A-Za-z].* # \S+\s*$", line):
        return []

    return line.split("#")[1].strip().split(",")
