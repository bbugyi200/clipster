# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

from setup_utils import install_requires, extras_require


setup(
    ### Project Description
    name="clipster",
    use_scm_version=True,
    url="https://github.com/mrichar1/clipster",
    license="LICENSE.md",
    author="Matthew Richardson",
    author_email="",
    description="python clipboard manager",
    long_description=__doc__,
    ### Package Requirements (i.e. dependencies)
    setup_requires=["setuptools_scm"],
    install_requires=install_requires(),
    extras_require=extras_require(),
    ### Package Contents
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    data_files=[
        ("share/licenses/clipster", ["LICENSE.md"]),
        ("share/doc/clipster", ["README.md"]),
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "clipster = clipster:main",
        ]
    },
    ### Platform / System Requirements
    platforms="any",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: X11 Applications",
        "Intended Audience :: End Users/Desktop",
        'License :: OSI Approved :: "License :: OSI Approved :: GNU Affero'
        " General Public License v3 or later (AGPLv3+)",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
