"""Helper functions used by setup.py."""

from pathlib import Path
import re
from typing import Dict, Iterator, List, Union


PathLike = Union[str, Path]


def install_requires() -> List[str]:
    return list(_requires("requirements.txt"))


def extras_require() -> Dict[str, List[str]]:
    result = {}
    reqtxt = "extra-requirements.txt"
    for extra in _collect_extras(reqtxt):
        result[extra] = list(_requires(reqtxt, extra=extra))
    return result


def _requires(reqtxt_basename: str, extra: str = None) -> Iterator[str]:
    reqtxt = Path(__file__).parent / reqtxt_basename
    reqs = reqtxt.read_text().split("\n")
    for req in reqs:
        if not req or req.lstrip().startswith(("#", "-")):
            continue

        req_and_comment = [x.strip() for x in req.split("#")]

        if extra is not None and (
            len(req_and_comment) == 1 or req_and_comment[1] != extra
        ):
            continue

        yield req_and_comment[0]


def _collect_extras(reqtxt: PathLike) -> Iterator[str]:
    reqtxt = Path(reqtxt)
    extra_set = set()
    for line in reqtxt.open():
        if re.search(r" # \S+\s*$", line):
            extra = line.split("#")[1].strip()
            if extra not in extra_set:
                yield extra
                extra_set.add(extra)
