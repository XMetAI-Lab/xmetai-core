#!/bin/bash
# Test runner script for xmetai-core
# Usage: bash scripts/run_tests.sh [target] [options]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo_warn "pytest not found. Installing test dependencies..."
    pip install pytest pytest-cov pytest-timeout pytest-xdist
fi

cd "$PROJECT_DIR"

# Default target
TARGET="${1:-fast}"
shift || true

case "$TARGET" in
    fast)
        echo_info "Running fast tests (no slow/gpu tests)..."
        pytest tests -v -m "not slow and not gpu" --timeout=60 "$@"
        ;;
    unit)
        echo_info "Running unit tests..."
        pytest tests/unit -v --timeout=60 "$@"
        ;;
    integration)
        echo_info "Running integration tests..."
        pytest tests/integration -v --timeout=120 "$@"
        ;;
    gpu)
        if ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null || ! python -c "import torch; assert torch.cuda.is_available()"; then
            echo_warn "CUDA not available, skipping GPU tests"
            exit 0
        fi
        echo_info "Running GPU tests..."
        pytest tests -v -m gpu --timeout=120 "$@"
        ;;
    numerical)
        echo_info "Running numerical stability tests..."
        pytest tests -v -m numerical --timeout=120 "$@"
        ;;
    all)
        echo_info "Running all tests (including slow/gpu)..."
        pytest tests -v --timeout=120 "$@"
        ;;
    coverage)
        echo_info "Running tests with coverage report..."
        pytest tests -v --cov=xmetai --cov-report=html --cov-report=term-missing --timeout=120 "$@"
        echo_success "Coverage report generated in htmlcov/index.html"
        ;;
    lint)
        echo_info "Running linters..."

        if command -v flake8 &> /dev/null; then
            echo_info "Running flake8..."
            flake8 xmetai --count --select=E9,F63,F7,F82 --show-source --statistics
        else
            echo_warn "flake8 not installed"
        fi

        if command -v pylint &> /dev/null; then
            echo_info "Running pylint..."
            pylint xmetai --disable=all --enable=E,F,W
        else
            echo_warn "pylint not installed"
        fi
        ;;
    ci)
        echo_info "Running CI test suite (CPU only)..."
        pytest tests -v -m "not gpu" --timeout=60 -q --maxfail=1 "$@"
        ;;
    *)
        echo "Usage: $0 [target] [pytest_options]"
        echo ""
        echo "Targets:"
        echo "  fast       - Fast tests (no slow/gpu) - DEFAULT"
        echo "  unit       - Unit tests only"
        echo "  integration - Integration tests only"
        echo "  gpu        - GPU tests (requires CUDA)"
        echo "  numerical  - Numerical stability tests"
        echo "  all        - All tests (including slow)"
        echo "  coverage   - Tests with coverage report"
        echo "  lint       - Run code linters"
        echo "  ci         - CI test suite (CPU only, fail on first error)"
        echo ""
        echo "Examples:"
        echo "  bash scripts/run_tests.sh fast"
        echo "  bash scripts/run_tests.sh unit -v --tb=short"
        echo "  bash scripts/run_tests.sh coverage"
        echo "  bash scripts/run_tests.sh ci --verbose"
        exit 1
        ;;
esac

echo_success "Tests completed!"

        echo_error "Unknown command: $COMMAND"
        $0 help
        exit 1
        ;;
esac
