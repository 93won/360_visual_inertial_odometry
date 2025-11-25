/**
 * @file      main.cpp
 * @brief     360-degree VIO demo
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "Frame.h"
#include "Feature.h"
#include "ConfigUtils.h"
#include "VizUtils.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

/**
 * @brief Get sorted list of image files from directory
 */
std::vector<std::string> GetImageFiles(const std::string& directory) {
    std::vector<std::string> image_files;
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <images_directory> [config_file]" << std::endl;
        std::cout << "Example: " << argv[0] << " ../datasets/seq_1/360-VIO_format/images ../config/default_config.yaml" << std::endl;
        return -1;
    }
    
    std::string images_dir = argv[1];
    std::string config_file = (argc > 2) ? argv[2] : "../config/default_config.yaml";
    
    // Load configuration
    auto& config = vio_360::ConfigUtils::GetInstance();
    if (!config.Load(config_file)) {
        std::cout << "Warning: Using default configuration values" << std::endl;
    }
    
    // Get image files
    std::vector<std::string> image_files = GetImageFiles(images_dir);
    if (image_files.empty()) {
        std::cerr << "Error: No images found in " << images_dir << std::endl;
        return -1;
    }
    
    std::cout << "Found " << image_files.size() << " images" << std::endl;
    
    // Read first image to get dimensions
    cv::Mat first_image = cv::imread(image_files[0], cv::IMREAD_GRAYSCALE);
    if (first_image.empty()) {
        std::cerr << "Error: Failed to read first image" << std::endl;
        return -1;
    }
    
    std::cout << "Original image size: " << first_image.cols << "x" << first_image.rows << std::endl;
    std::cout << "Tracking image size: " << config.camera_width << "x" << config.camera_height << std::endl;
    
    // Create estimator
    auto estimator = std::make_unique<vio_360::Estimator>();
    std::cout << "Initialized VIO estimator" << std::endl;
    
    // Create visualizer
    auto viz = std::make_unique<vio_360::VizUtils>(1920, 1080);
    viz->Initialize();
    std::cout << "Initialized visualizer" << std::endl;
    
    // Process frames
    double timestamp = 0.0;  // seconds
    const double dt = 1.0 / 30.0;  // 30 fps
    
    std::shared_ptr<vio_360::Frame> prev_frame = nullptr;
    
    for (size_t i = 0; i < image_files.size(); ++i) {
        // Check if should close
        if (viz->ShouldClose()) {
            break;
        }
        
        // Check if paused - skip processing but keep updating viewer
        if (viz->IsPaused()) {
            viz->Update(estimator.get(), estimator->GetCurrentFrame(), prev_frame);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Read image
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Warning: Failed to read " << image_files[i] << std::endl;
            continue;
        }
        
        // Resize if needed
        if (image.cols != config.camera_width || image.rows != config.camera_height) {
            cv::resize(image, image, cv::Size(config.camera_width, config.camera_height), 0, 0, cv::INTER_AREA);
        }
        
        // Process frame
        auto result = estimator->ProcessFrame(image, timestamp);
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        
        // Get current frame
        auto current_frame = estimator->GetCurrentFrame();
        
        // Update visualizer (viewer will draw tracking internally)
        viz->Update(estimator.get(), current_frame, prev_frame);
        
        // If initialization succeeded, pause automatically
        if (result.init_success) {
            viz->SetPaused(true);
            std::cout << "\n========================================" << std::endl;
            std::cout << "[MAIN] Auto-paused: Initialization complete!" << std::endl;
            std::cout << "       Press 'Pause' button to continue." << std::endl;
            std::cout << "========================================\n" << std::endl;
        }
        
        // Print progress
        if ((i + 1) % 10 == 0) {
            std::cout << "Frame " << std::setw(4) << (i + 1) 
                     << " | Features: " << std::setw(4) << result.num_features
                     << " | Tracked: " << std::setw(4) << result.num_tracked
                     << " | Init: " << (estimator->IsInitialized() ? "YES" : "NO ")
                     << " | " << std::fixed << std::setprecision(1) 
                     << (1000.0 / total_time) << " fps" << std::endl;
        }
        
        prev_frame = current_frame;
        timestamp += dt;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Processing Complete" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Total frames: " << estimator->GetAllFrames().size() << std::endl;
    std::cout << "Initialized: " << (estimator->IsInitialized() ? "YES" : "NO") << std::endl;
    
    return 0;
}
