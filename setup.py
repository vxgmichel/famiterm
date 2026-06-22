from setuptools import Extension, setup
from Cython.Build import cythonize


def get_extensions() -> list[Extension]:
    import numpy

    include_path: str = numpy.get_include()
    nescpu_extension = Extension(
        "famiterm.nescpu",
        include_dirs=[include_path],
        sources=["ext/nescpu.pyx"],
    )
    nesppu_extension = Extension(
        "famiterm.nesppu",
        include_dirs=[include_path],
        sources=["ext/nesppu.pyx"],
    )
    nesapu_extension = Extension(
        "famiterm.nesapu",
        include_dirs=[include_path],
        sources=["ext/nesapu.pyx"],
    )
    return cythonize(
        [
            nescpu_extension,
            nesppu_extension,
            nesapu_extension,
        ],
        compiler_directives={"language_level": "3"},
    )


setup(ext_modules=get_extensions())
