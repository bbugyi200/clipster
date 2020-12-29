# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Iterator, List

from setuptools import find_packages, setup


def install_requires() -> List[str]:
    return list(_requires("requirements.txt"))


def _requires(reqtxt_basename: str) -> Iterator[str]:
    reqtxt = Path(__file__).parent / reqtxt_basename
    reqs = reqtxt.read_text().split("\n")
    for req in reqs:
        if not req or req.lstrip().startswith(("#", "-")):
            continue
        yield req


setup(
    name="clipster",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    url="https://github.com/mrichar1/clipster",
    license="LICENSE.md",
    author="Matthew Richardson",
    author_email="",
    description="python clipboard manager",
    long_description=__doc__,
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    data_files=[
        ("share/licenses/clipster", ["LICENSE.md"]),
        ("share/doc/clipster", ["README.md"]),
    ],
    zip_safe=False,
    platforms="any",
    install_requires=install_requires(),
    entry_points={
        "console_scripts": [
            "clipster = clipster:main",
        ]
    },
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        "Development Status :: 5 - Production/Stable",
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        "Environment :: Console",
        "Environment :: X11 Applications",
        "Intended Audience :: End Users/Desktop",
        'License :: OSI Approved :: "License :: OSI Approved :: GNU Affero'
        " General Public License v3 or later (AGPLv3+)",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
)
