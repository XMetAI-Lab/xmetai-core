#!/bin/bash
# Quick test runner script for common workflows

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}$1${NC}"
}

echo_warn() {
    echo -e "${YELLOW}$1${NC}"
}

echo_error() {
    echo -e "${RED}$1${NC}"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo_error "pytest not found. Install with: pip install pytest"
    exit 1
fi

# Parse command line arguments
COMMAND=${1:-all}

case $COMMAND in
    all)
        echo_info "Running all tests..."
        pytest tests -v
        ;;
    cpu)
        echo_info "Running CPU tests..."
        pytest tests -v -m "not gpu"
        ;;
        gpu)
                echo_info "Running GPU tests with coverage (>=95%)..."
                pytest tests -v -m gpu \
                    --cov=xmetai.layers.rotary_attention --cov-report=term-missing --cov-fail-under=95 \
                    -W ignore::DeprecationWarning
                ;;
    unit)
        echo_info "Running unit tests..."
        pytest tests/test_layers_basic.py tests/test_layers_vision.py tests/test_models.py tests/test_loss_metrics.py tests/test_data.py -v
        ;;
    integration)
        echo_info "Running integration tests..."
        pytest tests/test_integration.py -v
        ;;
    numerical)
        echo_info "Running numerical stability tests..."
        pytest tests -v -m numerical
        ;;
    coverage)
        echo_info "Running tests with coverage report..."
        pytest tests -v --cov=xmetai --cov-report=html --cov-report=term-missing
        echo_info "Coverage report generated in htmlcov/index.html"
        ;;
    fast)
        echo_info "Running fast tests (no slow/gpu tests)..."
        pytest tests -v -m "not slow and not gpu"
        ;;
    lint)
        echo_info "Running linters..."
        if command -v flake8 &> /dev/null; then
            flake8 xmetai tests --max-line-length=120
        fi
        if command -v black &> /dev/null; then
            black --check xmetai tests
        fi
        if command -v isort &> /dev/null; then
            isort --check-only xmetai tests
        fi
        ;;
    help)
        cat << EOF
Test Runner Commands:

  all          - Run all tests
  cpu          - Run CPU tests (skip GPU)
  gpu          - Run GPU tests (requires CUDA)
  unit         - Run unit tests only
  integration  - Run integration tests
  numerical    - Run numerical stability tests
  coverage     - Run tests with coverage report (generates htmlcov/)
  fast         - Run fast tests (excludes slow/GPU tests)
  lint         - Run code quality checks

Examples:
  ./run_tests.sh                    # Run all tests
  ./run_tests.sh cpu               # Run CPU tests
  ./run_tests.sh coverage           # Generate coverage report
  ./run_tests.sh fast -- -x         # Run fast tests, stop on first failure

Options can be passed to pytest after --:
  ./run_tests.sh unit -- -k layer   # Run unit tests matching 'layer'
  ./run_tests.sh all -- -x          # Stop on first failure
EOF
        ;;
    *)
        echo_error "Unknown command: $COMMAND"
        $0 help
        exit 1
        ;;
esac
