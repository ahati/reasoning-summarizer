#!/bin/bash
# build-llama.sh - Fetches and builds llama.cpp for CGo bindings
#
# This script:
# 1. Clones llama.cpp (shallow) into llama.cpp/
# 2. Builds the static libraries needed for CGo
#
# Usage:
#   ./build-llama.sh [version]
#
# Arguments:
#   version - Optional llama.cpp version/commit/tag (default: b8508)
#
# Environment variables:
#   LLAMA_CPP_VERSION - Override the default version
#   LLAMA_CPP_REPO    - Override the git repository URL
#   CMAKE_BUILD_TYPE  - Build type (default: Release)
#
# After running this script, go build will work.

set -e

# Configuration
LLAMA_VERSION="${LLAMA_CPP_VERSION:-${1:-b8508}}"
LLAMA_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$ROOT_DIR/llama.cpp"

echo "=== Fetching llama.cpp ==="
echo "Version: $LLAMA_VERSION"
echo "Repository: $LLAMA_REPO"
echo "Target: $LLAMA_DIR"

# Check if llama.cpp already exists with correct version
if [ -d "$LLAMA_DIR/.git" ]; then
    CURRENT_VERSION=$(cd "$LLAMA_DIR" && git describe --tags --always 2>/dev/null || git rev-parse --short HEAD)
    echo "Existing llama.cpp found (version: $CURRENT_VERSION)"

    # Check if it's the version we want
    if [ "$CURRENT_VERSION" = "$LLAMA_VERSION" ] || [ "$(cd "$LLAMA_DIR" && git rev-parse HEAD)" = "$(cd "$LLAMA_DIR" && git rev-parse "$LLAMA_VERSION" 2>/dev/null || echo '')" ]; then
        echo "Correct version already checked out"
    else
        echo "Updating to version $LLAMA_VERSION..."
        cd "$LLAMA_DIR"
        git fetch --depth 1 origin "$LLAMA_VERSION" 2>/dev/null || git fetch origin
        git checkout "$LLAMA_VERSION"
    fi
else
    # Remove incomplete/stub directory if it exists
    if [ -d "$LLAMA_DIR" ]; then
        echo "Removing existing llama.cpp directory..."
        rm -rf "$LLAMA_DIR"
    fi

    echo "Cloning llama.cpp..."
    git clone --depth 1 --branch "$LLAMA_VERSION" "$LLAMA_REPO" "$LLAMA_DIR"
fi

# Build static libraries
echo ""
echo "=== Building llama.cpp static libraries ==="
BUILD_DIR="$LLAMA_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with CMake..."
cmake .. \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) llama ggml ggml-base ggml-cpu

# Verify libraries exist
LIBRARIES=(
    "$BUILD_DIR/src/libllama.a"
    "$BUILD_DIR/ggml/src/libggml.a"
    "$BUILD_DIR/ggml/src/libggml-base.a"
    "$BUILD_DIR/ggml/src/libggml-cpu.a"
)

echo ""
echo "=== Verifying static libraries ==="
MISSING=0
for LIB in "${LIBRARIES[@]}"; do
    if [ -f "$LIB" ]; then
        SIZE=$(stat -f%z "$LIB" 2>/dev/null || stat -c%s "$LIB" 2>/dev/null || echo "unknown")
        echo "OK: $LIB ($SIZE bytes)"
    else
        echo "MISSING: $LIB"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Some static libraries are missing!"
    echo "The build may have failed. Check the CMake output above."
    exit 1
fi

echo ""
echo "=== Build complete ==="
echo "Headers: $LLAMA_DIR/include/"
echo "Libraries: $BUILD_DIR/"
echo ""
echo "You can now run: go build ./..."