from setuptools import setup

setup(
    name="affinegaps",
    version="0.2.4",
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    description="JIT-compiled string edit-distances including Needleman-Wunsch, Smith-Waterman, Wagner-Fisher, and Gotoh algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/unum-bio/AffineGaps",
    py_modules=["affinegaps"],
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    extras_require={
        "jit": ["numba", "colorama"],
        "dev": ["biopython", "stringzilla", "pytest", "pytest-repeat"],
    },
    entry_points={
        "console_scripts": [
            "affinegaps=affinegaps:main",
        ],
    },
)
