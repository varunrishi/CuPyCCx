"""
Builds the C++ GoogleTest binary before the test session if not already built.
"""

import os
import subprocess
import pytest

BUILD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build_cpp_tests")
)
EIGEN_DIR  = "/usr/local/Cellar/eigen/3.4.0_1/share/eigen3/cmake"
CPP_BIN    = os.path.join(BUILD_DIR, "tests", "cpp", "test_cupyccx")
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    subprocess.run(
        [
            "cmake", SOURCE_DIR,
            "-DCUPYCCX_BUILD_TESTS=ON",
            "-DCUPYCCX_BUILD_PYTHON=OFF",
            f"-DEigen3_DIR={EIGEN_DIR}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=BUILD_DIR, check=True, capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", ".", "--parallel"],
        cwd=BUILD_DIR, check=True, capture_output=True,
    )


def pytest_configure(config):
    """Build C++ tests once at session start if the binary is missing."""
    if not os.path.exists(CPP_BIN):
        try:
            _build()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # test_cpp.py will skip gracefully if binary is absent
