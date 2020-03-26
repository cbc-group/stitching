"""
Minimal setup.py to simplify project setup.
"""
from setuptools import find_packages, setup

setup(
    name="stitching",
    version="0.1",
    description="",
    author="Liu, Yen-Ting",
    author_email="ytliu@gate.sinica.edu.tw",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={"": ["data/*"]},
    python_requires="~=3.7",
    install_requires=[
        "click",
        "coloredlogs",
        "dask~=2.12.0",
        "distributed~=2.12.0",
        "imageio",
        "numpy",
        "pandas",
        "pyyaml",
        "scipy",
        "tqdm",
        "utoolbox>=0.6.0",
    ],
    zip_safe=True,
    extras_require={"viewer": ["pyside2", "pyqtgraph>=0.11.0rc0"]},
    entry_points={
        "console_scripts": [
            "preview=stitching.preview:main",
            "archive=stitching.convert:main",
        ]
    },
)
