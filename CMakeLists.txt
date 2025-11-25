cmake_minimum_required(VERSION 3.10)
project(vio_360)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release mode optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/database
    ${CMAKE_CURRENT_SOURCE_DIR}/src/processing
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util
)

# Source files
set(SOURCES
    src/database/Camera.cpp
    src/database/Feature.cpp
    src/database/Frame.cpp
    src/processing/FeatureTracker.cpp
    src/processing/Estimator.cpp
    src/util/LieUtils.cpp
    src/util/ConfigUtils.cpp
    src/util/VizUtils.cpp
)

# Create library
add_library(vio_360_lib ${SOURCES})
target_link_libraries(vio_360_lib
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)

# Main application
add_executable(feature_tracking_demo
    app/main.cpp
)

target_link_libraries(feature_tracking_demo
    vio_360_lib
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)

# Set output directory
set_target_properties(feature_tracking_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Pangolin include: ${Pangolin_INCLUDE_DIRS}")
