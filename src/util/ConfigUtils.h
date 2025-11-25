/**
 * @file      ConfigUtils.h
 * @brief     Configuration utility for loading YAML config files
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <memory>

namespace vio_360 {

/**
 * @brief Singleton configuration class
 */
class ConfigUtils {
public:
    // Get singleton instance
    static ConfigUtils& GetInstance();
    
    // Load configuration from YAML file
    bool Load(const std::string& config_file);
    
    // Camera parameters
    int camera_width;
    int camera_height;
    float camera_polar_exclusion_ratio;
    int camera_boundary_margin;
    
    // Feature detection
    int max_features;
    double quality_level;
    double min_distance;
    
    // Grid parameters
    int grid_cols;
    int grid_rows;
    int max_features_per_grid;
    
    // Optical flow
    int optical_flow_window_size;
    int optical_flow_max_level;
    int optical_flow_max_iterations;
    double optical_flow_epsilon;
    
    // RANSAC
    int ransac_max_iterations;
    float ransac_threshold_degrees;
    float ransac_confidence;
    
    // Tracking
    float tracking_min_features_ratio;
    float tracking_min_parallax_for_keyframe;
    
    // Initialization
    int initialization_window_size;
    float initialization_min_parallax;
    int initialization_min_features;
    int initialization_min_observations;
    int initialization_grid_cols;
    int initialization_grid_rows;
    float initialization_ransac_threshold;
    int initialization_ransac_iterations;
    float initialization_min_inlier_ratio;
    float initialization_max_reprojection_error;
    
    // Visualization
    float visualization_scale;
    bool visualization_show_grid;
    cv::Scalar visualization_grid_color;
    int visualization_grid_thickness;
    int visualization_max_age_for_color;
    int visualization_stable_age_threshold;
    bool visualization_highlight_clustered_grid;
    cv::Scalar visualization_clustered_grid_color;
    float visualization_clustered_std_ratio;
    
    // Video output
    float video_output_fps;
    std::string video_output_codec;
    
    // Camera extrinsics
    Eigen::Matrix4f T_BC;
    
    // IMU parameters (for future use)
    float imu_frequency;
    float imu_gyro_noise;
    float imu_accel_noise;
    float imu_gyro_bias_noise;
    float imu_accel_bias_noise;
    
private:
    ConfigUtils();
    ~ConfigUtils() = default;
    
    // Disable copy and move
    ConfigUtils(const ConfigUtils&) = delete;
    ConfigUtils& operator=(const ConfigUtils&) = delete;
    ConfigUtils(ConfigUtils&&) = delete;
    ConfigUtils& operator=(ConfigUtils&&) = delete;
    
    // Helper functions
    void SetDefaultValues();
    Eigen::Matrix4f LoadMatrix4f(const cv::FileNode& node);
};

} // namespace vio_360
