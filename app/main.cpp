/**
 * @file      main.cpp
 * @brief     360-degree VIO demo
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
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
#include "Logger.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

std::vector<double> LoadCameraTimestamps(const std::string& filepath) {
    std::vector<double> timestamps;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        LOG_ERROR("Failed to open {}", filepath);
        return timestamps;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            timestamps.push_back(std::stod(line));
        } catch (...) {}
    }
    
    return timestamps;
}

std::vector<vio_360::IMUData> LoadIMUData(const std::string& filepath) {
    std::vector<vio_360::IMUData> imu_data;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        LOG_ERROR("Failed to open {}", filepath);
        return imu_data;
    }
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, item, ',')) {
            tokens.push_back(item);
        }
        
        if (tokens.size() != 7) continue;
        
        try {
            vio_360::IMUData imu;
            imu.timestamp = std::stod(tokens[0]);
            imu.ax = std::stof(tokens[1]);
            imu.ay = std::stof(tokens[2]);
            imu.az = std::stof(tokens[3]);
            imu.gx = std::stof(tokens[4]);
            imu.gy = std::stof(tokens[5]);
            imu.gz = std::stof(tokens[6]);
            imu_data.push_back(imu);
        } catch (...) {}
    }
    
    return imu_data;
}

std::vector<vio_360::IMUData> GetIMUBetweenTimestamps(
    const std::vector<vio_360::IMUData>& all_imu_data,
    double start_time,
    double end_time
) {
    std::vector<vio_360::IMUData> filtered_imu;
    for (const auto& imu : all_imu_data) {
        if (imu.timestamp >= start_time && imu.timestamp < end_time) {
            filtered_imu.push_back(imu);
        }
    }
    return filtered_imu;
}

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
    vio_360::Logger::Init();
    
    if (argc < 2) {
        LOG_INFO("Usage: {} <dataset_directory> [config_file]", argv[0]);
        return -1;
    }
    
    std::string dataset_dir = argv[1];
    std::string config_file = (argc > 2) ? argv[2] : "../config/default_config.yaml";
    
    if (dataset_dir.back() != '/') dataset_dir += '/';
    
    // Load configuration
    auto& config = vio_360::ConfigUtils::GetInstance();
    config.Load(config_file);
    
    // Load data
    auto cam_timestamps = LoadCameraTimestamps(dataset_dir + "cam_timestamps.txt");
    auto all_imu_data = LoadIMUData(dataset_dir + "imu_data.csv");
    auto image_files = GetImageFiles(dataset_dir + "images/");
    
    if (cam_timestamps.empty() || all_imu_data.empty() || image_files.empty()) {
        LOG_ERROR("Failed to load dataset");
        return -1;
    }
    
    // Adjust sizes if needed
    size_t min_count = std::min(image_files.size(), cam_timestamps.size());
    image_files.resize(min_count);
    cam_timestamps.resize(min_count);
    
    // Calculate IMU frequency
    double imu_duration = all_imu_data.back().timestamp - all_imu_data.front().timestamp;
    double imu_freq = all_imu_data.size() / imu_duration;
    
    LOG_INFO("Dataset: {} images, {} IMU ({:.0f}Hz), {:.1f}s duration",
             image_files.size(), all_imu_data.size(), imu_freq,
             cam_timestamps.back() - cam_timestamps.front());
    
    // Read first image
    cv::Mat first_image = cv::imread(image_files[0], cv::IMREAD_GRAYSCALE);
    if (first_image.empty()) {
        LOG_ERROR("Failed to read first image");
        return -1;
    }
    
    // Create estimator and visualizer
    auto estimator = std::make_unique<vio_360::Estimator>();
    auto viz = std::make_unique<vio_360::VizUtils>(1920, 1080);
    viz->Initialize();
    
    LOG_INFO("Starting VIO...");
    
    std::shared_ptr<vio_360::Frame> prev_frame = nullptr;
    double prev_timestamp = 0.0;
    
    for (size_t i = 0; i < image_files.size(); ++i) {
        if (viz->ShouldClose()) break;
        
        // Handle pause and step
        if (viz->IsPaused()) {
            if (!viz->StepRequested()) {
                viz->Update(estimator.get(), estimator->GetCurrentFrame(), prev_frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                --i;  // Stay on current frame
                continue;
            }
            // Step requested - proceed with this frame
        }
        
        double current_timestamp = cam_timestamps[i];
        
        std::vector<vio_360::IMUData> frame_imu_data;
        if (i > 0) {
            frame_imu_data = GetIMUBetweenTimestamps(all_imu_data, prev_timestamp, current_timestamp);
        }
        
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        
        if (image.cols != config.camera_width || image.rows != config.camera_height) {
            cv::resize(image, image, cv::Size(config.camera_width, config.camera_height), 0, 0, cv::INTER_AREA);
        }
        
        vio_360::Estimator::EstimationResult result;
        if (i > 0 && !frame_imu_data.empty()) {
            result = estimator->ProcessFrame(image, current_timestamp, frame_imu_data);
        } else {
            result = estimator->ProcessFrame(image, current_timestamp);
        }
        
        auto current_frame = estimator->GetCurrentFrame();
        viz->Update(estimator.get(), current_frame, prev_frame);
        
        if (result.init_success) {
            viz->SetPaused(true);
            LOG_INFO("Initialization complete! Press 'Pause' to continue.");
        }
        
        prev_frame = current_frame;
        prev_timestamp = current_timestamp;
    }
    
    LOG_INFO("Processing complete: {} frames, initialized={}",
             estimator->GetAllFrames().size(),
             estimator->IsInitialized() ? "yes" : "no");
    
    viz->SetPaused(true);
    while (!viz->ShouldClose()) {
        viz->Update(estimator.get(), estimator->GetCurrentFrame(), prev_frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    return 0;
}
