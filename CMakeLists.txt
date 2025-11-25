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

# Set OpenGL preference
set(OpenGL_GL_PREFERENCE GLVND)

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# Add thirdparty libraries
set(BUILD_PANGOLIN_EXAMPLES OFF CACHE BOOL "Build Pangolin examples" FORCE)
set(BUILD_PANGOLIN_TESTS OFF CACHE BOOL "Build Pangolin tests" FORCE)
set(BUILD_PANGOLIN_TOOLS OFF CACHE BOOL "Build Pangolin tools" FORCE)
set(BUILD_PANGOLIN_PYTHON OFF CACHE BOOL "Build Pangolin Python bindings" FORCE)
add_subdirectory(thirdparty/pangolin EXCLUDE_FROM_ALL)

set(BUILD_EXAMPLES OFF CACHE BOOL "Build Ceres examples" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build Ceres benchmarks" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build Ceres tests" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable Ceres uninstall target" FORCE)
add_subdirectory(thirdparty/ceres-solver EXCLUDE_FROM_ALL)

# spdlog (header-only)
add_subdirectory(thirdparty/spdlog EXCLUDE_FROM_ALL)

# Include directories - thirdparty Pangolin ONLY (not system)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pangolin/include
    ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/pangolin/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/include
    ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/ceres-solver/config
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/database
    ${CMAKE_CURRENT_SOURCE_DIR}/src/processing
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optimization
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util
)

# Source files
set(SOURCES
    src/database/Camera.cpp
    src/database/Feature.cpp
    src/database/Frame.cpp
    src/database/MapPoint.cpp
    src/processing/FeatureTracker.cpp
    src/processing/Estimator.cpp
    src/processing/Initializer.cpp
    src/processing/IMUPreintegrator.cpp
    src/optimization/Factors.cpp
    src/optimization/Parameters.cpp
    src/optimization/Optimizer.cpp
    src/util/LieUtils.cpp
    src/util/ConfigUtils.cpp
    src/util/VizUtils.cpp
)

# Create library
add_library(vio_360_lib ${SOURCES})
target_link_libraries(vio_360_lib
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    pango_display
    pango_opengl
    pango_windowing
    pango_vars
    pango_core
    ceres
    spdlog::spdlog_header_only
)

# Main application
add_executable(feature_tracking_demo
    app/main.cpp
)

target_link_libraries(feature_tracking_demo
    vio_360_lib
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    pango_display
    pango_opengl
    pango_windowing
    pango_vars
    pango_core
    ceres
    spdlog::spdlog_header_only
)

# Set output directory
set_target_properties(feature_tracking_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIR}")
