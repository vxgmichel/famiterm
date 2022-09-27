import os
from pathlib import Path
from setuptools import Extension, setup  # type: ignore

# Read the contents of the README file
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text()


# Defer call to `numpy.get_include`
class NumpyIncludePath(os.PathLike):
    def __str__(self) -> str:
        return self.__fspath__()

    def __fspath__(self) -> str:
        import numpy

        include_path: str = numpy.get_include()
        return os.fspath(include_path)


# Extensions
nescpu_extension = Extension(
    "famiterm.nescpu",
    include_dirs=[NumpyIncludePath()],
    sources=["ext/nescpu.pyx"],
)
nesppu_extension = Extension(
    "famiterm.nesppu",
    include_dirs=[NumpyIncludePath()],
    sources=["ext/nesppu.pyx"],
)
nesapu_extension = Extension(
    "famiterm.nesapu",
    include_dirs=[NumpyIncludePath()],
    sources=["ext/nesapu.pyx"],
)


setup(
    name="famiterm",
    version="0.1.1",
    packages=["famiterm"],
    setup_requires=["setuptools>=42", "Cython>=0.29.13", "numpy"],
    ext_modules=[
        nescpu_extension,
        nesppu_extension,
        nesapu_extension,
    ],
    install_requires=["numpy>=1.20", "gambaterm>=0.12.0"],
    extras_require={
        "controller-support": ["pygame>=1.9.5"],
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "famiterm = famiterm:main",
            "famiterm-ssh = famiterm.ssh:main",
        ],
    },
    package_data={"famiterm": ["py.typed"]},
    description="A NES emulator running in the terminal",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/vxgmichel/famiterm",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    author="Vincent Michel",
    author_email="vxgmichel@gmail.com",
)
