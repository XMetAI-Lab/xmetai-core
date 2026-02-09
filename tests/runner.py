"""
Test runner and utilities for the xmetai test suite.
Provides convenient commands to run different test categories.
"""

import subprocess
import sys
from pathlib import Path


def run_tests(args=None):
    """Run pytest with given arguments."""
    cmd = ["python", "-m", "pytest"]
    if args:
        cmd.extend(args)
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def run_all_tests():
    """Run all tests."""
    print("Running all tests...")
    return run_tests(["-v"])


def run_cpu_tests():
    """Run only CPU tests."""
    print("Running CPU tests...")
    return run_tests(["-v", "-m", "not gpu"])


def run_gpu_tests():
    """Run only GPU tests (requires CUDA)."""
    print("Running GPU tests...")
    return run_tests(["-v", "-m", "gpu"])


def run_unit_tests():
    """Run unit tests (exclude slow/integration tests)."""
    print("Running unit tests...")
    return run_tests(
        [
            "-v",
            "-m",
            "not slow",
            "tests/unit/test_layers_basic.py",
            "tests/unit/test_layers_vision.py",
            "tests/unit/test_models.py",
            "tests/unit/test_loss_metrics.py",
            "tests/unit/test_data.py",
            "tests/unit/test_graphcast_grid.py",
            "tests/unit/test_graphcast_dataset.py",
            "tests/unit/test_graphcast_model.py",
            "tests/unit/test_lora.py",
        ]
    )


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    return run_tests(["-v", "tests/integration/test_integration.py"])


def run_numerical_tests():
    """Run numerical stability tests."""
    print("Running numerical stability tests...")
    return run_tests(["-v", "-m", "numerical"])


def run_with_coverage():
    """Run tests with coverage report."""
    print("Running tests with coverage...")
    return run_tests(
        [
            "-v",
            "--cov=xmetai",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "all":
            sys.exit(run_all_tests().returncode)
        elif cmd == "cpu":
            sys.exit(run_cpu_tests().returncode)
        elif cmd == "gpu":
            sys.exit(run_gpu_tests().returncode)
        elif cmd == "unit":
            sys.exit(run_unit_tests().returncode)
        elif cmd == "integration":
            sys.exit(run_integration_tests().returncode)
        elif cmd == "numerical":
            sys.exit(run_numerical_tests().returncode)
        elif cmd == "coverage":
            sys.exit(run_with_coverage().returncode)
        else:
            print(f"Unknown command: {cmd}")
            print("\nUsage:")
            print("  python -m tests.runner [command]")
            print("\nCommands:")
            print("  all         - Run all tests")
            print("  cpu         - Run CPU tests only")
            print("  gpu         - Run GPU tests only")
            print("  unit        - Run unit tests")
            print("  integration - Run integration tests")
            print("  numerical   - Run numerical stability tests")
            print("  coverage    - Run tests with coverage report")
            sys.exit(1)
    else:
        # Default: run all tests
        sys.exit(run_all_tests().returncode)
