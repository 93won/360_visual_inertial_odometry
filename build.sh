#!/bin/bash

# Build script for 360 VIO

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
