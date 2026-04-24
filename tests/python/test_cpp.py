"""
Runs the compiled C++ GoogleTest suite as individual pytest items.
Skipped entirely if the binary hasn't been built.
"""

import subprocess
import pytest
import os
CPP_BIN = os.path.join(
    os.path.dirname(__file__), "..", "..", "build_cpp_tests", "tests", "cpp", "test_cupyccx"
)
CPP_BIN = os.path.abspath(CPP_BIN)


def _list_cpp_tests():
    result = subprocess.run(
        [CPP_BIN, "--gtest_list_tests"],
        capture_output=True, text=True,
    )
    tests = []
    suite = None
    for line in result.stdout.splitlines():
        if line.endswith("."):
            suite = line.strip()
        elif suite and line.startswith("  "):
            tests.append(f"{suite}{line.strip()}")
    return tests


_HAS_BIN = pytest.mark.skipif(
    not __import__("os").path.exists(CPP_BIN),
    reason="C++ test binary not built",
)

try:
    _CPP_TESTS = _list_cpp_tests() if __import__("os").path.exists(CPP_BIN) else []
except Exception:
    _CPP_TESTS = []


@_HAS_BIN
@pytest.mark.parametrize("test_id", _CPP_TESTS)
def test_cpp(test_id):
    result = subprocess.run(
        [CPP_BIN, f"--gtest_filter={test_id}", "--gtest_color=no"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
