#!/bin/bash

# Build script for 360 VIO

# Check and install dependencies
install_deps() {
    echo "Checking dependencies..."
    
    DEPS=""
    
    # Check glog
    if ! pkg-config --exists libglog 2>/dev/null && [ ! -f /usr/include/glog/logging.h ]; then
        DEPS="$DEPS libgoogle-glog-dev"
    fi
    
    # Check gflags
    if ! pkg-config --exists gflags 2>/dev/null && [ ! -f /usr/include/gflags/gflags.h ]; then
        DEPS="$DEPS libgflags-dev"
    fi
    
    # Check SuiteSparse (for Ceres)
    if [ ! -f /usr/include/suitesparse/cholmod.h ] && [ ! -f /usr/include/cholmod.h ]; then
        DEPS="$DEPS libsuitesparse-dev"
    fi
    
    # Check BLAS/LAPACK
    if [ ! -f /usr/lib/x86_64-linux-gnu/libblas.so ] && [ ! -f /usr/lib/libblas.so ]; then
        DEPS="$DEPS libblas-dev liblapack-dev"
    fi
    
    if [ -n "$DEPS" ]; then
        echo "Installing missing dependencies:$DEPS"
        sudo apt-get update
        sudo apt-get install -y $DEPS
    else
        echo "All dependencies found."
    fi
}

install_deps

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Build successful!"
    echo "==================================="
    echo ""
    echo "Run the demo with:"
    echo "  ./bin/feature_tracking_demo <images_directory> [output_video]"
    echo ""
    echo "Example:"
    echo "  ./bin/feature_tracking_demo ../datasets/seq_1/360-VIO_format/images tracking_output.mp4"
    echo ""
else
    echo ""
    echo "==================================="
    echo "Build failed!"
    echo "==================================="
    exit 1
fi
