#!/bin/bash
# build-llama.sh - Builds the llama.cpp submodule for CGo bindings
#
# This script builds the llama.cpp submodule that's already present in the repo.
# llama.cpp is included as a git submodule, so it's automatically fetched with the repo.
#
# Usage:
#   ./build-llama.sh
#
# Environment variables:
#   CMAKE_BUILD_TYPE - Build type (default: Release)
#
# The script creates:
#   - llama.cpp/build/ directory with compiled static libraries

set -e

# Configuration
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$ROOT_DIR/llama.cpp"

echo "=== Building llama.cpp submodule ==="
echo "Source: $LLAMA_DIR"
echo "Build type: $BUILD_TYPE"

# Verify submodule exists
if [ ! -d "$LLAMA_DIR" ]; then
    echo "ERROR: llama.cpp submodule not found!"
    echo "Make sure to clone with --recurse-submodules or run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "$LLAMA_DIR/CMakeLists.txt" ]; then
    echo "ERROR: llama.cpp submodule appears empty!"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Build static libraries
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