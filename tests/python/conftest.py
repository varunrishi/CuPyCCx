"""
Builds the C++ GoogleTest binary before the test session if not already built.
"""

import os
import subprocess
import pytest

BUILD_DIR  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build_cpp_tests")
)
CPP_BIN    = os.path.join(BUILD_DIR, "tests", "cpp", "test_cupyccx")
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Eigen3 cmake config: prefer env var, then common platform paths
_EIGEN3_CANDIDATES = [
    os.environ.get("EIGEN3_DIR", ""),
    "/usr/share/eigen3/cmake",                               # Ubuntu apt
    "/usr/local/Cellar/eigen/3.4.0_1/share/eigen3/cmake",   # macOS Homebrew
    "/opt/homebrew/share/eigen3/cmake",                      # macOS arm Homebrew
]
EIGEN_DIR = next((p for p in _EIGEN3_CANDIDATES if p and os.path.isdir(p)), "")


def _build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    cmake_args = [
        "cmake", SOURCE_DIR,
        "-DCUPYCCX_BUILD_TESTS=ON",
        "-DCUPYCCX_BUILD_PYTHON=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if EIGEN_DIR:
        cmake_args.append(f"-DEigen3_DIR={EIGEN_DIR}")
    subprocess.run(cmake_args, cwd=BUILD_DIR, check=True, capture_output=True)
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
