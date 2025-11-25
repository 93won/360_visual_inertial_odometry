/**
 * @file      ConfigUtils.cpp
 * @brief     Implementation of configuration utility
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "ConfigUtils.h"
#include "Logger.h"

namespace vio_360 {

ConfigUtils& ConfigUtils::GetInstance() {
    static ConfigUtils instance;
    return instance;
}

ConfigUtils::ConfigUtils() {
    SetDefaultValues();
}

void ConfigUtils::SetDefaultValues() {
    // Camera parameters
    camera_width = 960;
    camera_height = 480;
    camera_polar_exclusion_ratio = 0.15f;
    camera_boundary_margin = 20;
    
    // Feature detection
    max_features = 1000;
    quality_level = 0.01;
    min_distance = 30.0;
    
    // Grid parameters
    grid_cols = 20;
    grid_rows = 10;
    max_features_per_grid = 4;
    
    // Optical flow
    optical_flow_window_size = 21;
    optical_flow_max_level = 3;
    optical_flow_max_iterations = 30;
    optical_flow_epsilon = 0.01;
    
    // RANSAC
    ransac_max_iterations = 1000;
    ransac_threshold_degrees = 2.0f;
    ransac_confidence = 0.99f;
    
    // Tracking
    tracking_min_features_ratio = 0.5f;
    tracking_min_parallax_for_keyframe = 10.0f;
    
    // Initialization
    initialization_window_size = 20;
    initialization_min_parallax = 10.0f;
    initialization_min_features = 50;
    initialization_min_observations = 10;
    initialization_grid_cols = 8;
    initialization_grid_rows = 4;
    initialization_ransac_threshold = 0.001f;
    initialization_ransac_iterations = 200;
    initialization_min_inlier_ratio = 0.7f;
    initialization_max_reprojection_error = 2.0f;
    
    // Visualization
    visualization_scale = 1.0f;
    visualization_show_grid = true;
    visualization_grid_color = cv::Scalar(100, 255, 100);  // Light green
    visualization_grid_thickness = 1;
    visualization_max_age_for_color = 10;
    visualization_stable_age_threshold = 5;
    visualization_highlight_clustered_grid = true;
    visualization_clustered_grid_color = cv::Scalar(255, 100, 100);  // Light blue
    visualization_clustered_std_ratio = 0.15f;  // 15% of grid size
    
    // Video output
    video_output_fps = 30.0f;
    video_output_codec = "mp4v";
    
    // Camera extrinsics (identity by default)
    T_BC = Eigen::Matrix4f::Identity();
    
    // IMU parameters
    imu_frequency = 200.0f;
    imu_gyro_noise = 0.001f;
    imu_accel_noise = 0.01f;
    imu_gyro_bias_noise = 0.0001f;
    imu_accel_bias_noise = 0.001f;
}

bool ConfigUtils::Load(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        LOG_WARN("Config file not found, using defaults");
        return false;
    }
    
    // Camera parameters
    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        camera_width = (int)camera["width"];
        camera_height = (int)camera["height"];
        camera_polar_exclusion_ratio = (float)(double)camera["polar_exclusion_ratio"];
        camera_boundary_margin = (int)camera["boundary_margin"];
    }
    
    // Feature detection
    cv::FileNode feature_detection = fs["feature_detection"];
    if (!feature_detection.empty()) {
        max_features = (int)feature_detection["max_features"];
        quality_level = (double)feature_detection["quality_level"];
        min_distance = (double)feature_detection["min_distance"];
        grid_cols = (int)feature_detection["grid_cols"];
        grid_rows = (int)feature_detection["grid_rows"];
        max_features_per_grid = (int)feature_detection["max_features_per_grid"];
    }
    
    // Optical flow
    cv::FileNode optical_flow = fs["optical_flow"];
    if (!optical_flow.empty()) {
        optical_flow_window_size = (int)optical_flow["window_size"];
        optical_flow_max_level = (int)optical_flow["max_level"];
        optical_flow_max_iterations = (int)optical_flow["max_iterations"];
        optical_flow_epsilon = (double)optical_flow["epsilon"];
    }
    
    // RANSAC
    cv::FileNode ransac = fs["ransac"];
    if (!ransac.empty()) {
        ransac_max_iterations = (int)ransac["max_iterations"];
        ransac_threshold_degrees = (float)(double)ransac["threshold_degrees"];
        ransac_confidence = (float)(double)ransac["confidence"];
    }
    
    // Tracking
    cv::FileNode tracking = fs["tracking"];
    if (!tracking.empty()) {
        tracking_min_features_ratio = (float)(double)tracking["min_features_ratio"];
        if (!tracking["min_parallax_for_keyframe"].empty()) {
            tracking_min_parallax_for_keyframe = (float)(double)tracking["min_parallax_for_keyframe"];
        }
    }
    
    // Initialization
    cv::FileNode initialization = fs["initialization"];
    if (!initialization.empty()) {
        initialization_window_size = (int)initialization["window_size"];
        initialization_min_parallax = (float)(double)initialization["min_parallax"];
        initialization_min_features = (int)initialization["min_features"];
        initialization_min_observations = (int)initialization["min_observations"];
        initialization_grid_cols = (int)initialization["grid_cols"];
        initialization_grid_rows = (int)initialization["grid_rows"];
        initialization_ransac_threshold = (float)(double)initialization["ransac_threshold"];
        initialization_ransac_iterations = (int)initialization["ransac_iterations"];
        initialization_min_inlier_ratio = (float)(double)initialization["min_inlier_ratio"];
        initialization_max_reprojection_error = (float)(double)initialization["max_reprojection_error"];
    }
    
    // Visualization
    cv::FileNode visualization = fs["visualization"];
    if (!visualization.empty()) {
        visualization_scale = (float)(double)visualization["scale"];
        visualization_show_grid = (int)visualization["show_grid"] != 0;
        
        cv::FileNode grid_color = visualization["grid_color"];
        if (grid_color.isSeq() && grid_color.size() == 3) {
            visualization_grid_color = cv::Scalar(
                (int)grid_color[0],
                (int)grid_color[1],
                (int)grid_color[2]
            );
        }
        
        visualization_grid_thickness = (int)visualization["grid_thickness"];
        visualization_max_age_for_color = (int)visualization["max_age_for_color"];
        visualization_stable_age_threshold = (int)visualization["stable_age_threshold"];
        
        visualization_highlight_clustered_grid = (int)visualization["highlight_clustered_grid"] != 0;
        
        cv::FileNode clustered_color = visualization["clustered_grid_color"];
        if (clustered_color.isSeq() && clustered_color.size() == 3) {
            visualization_clustered_grid_color = cv::Scalar(
                (int)clustered_color[0],
                (int)clustered_color[1],
                (int)clustered_color[2]
            );
        }
        
        visualization_clustered_std_ratio = (float)(double)visualization["clustered_std_ratio"];
    }
    
    // Video output
    cv::FileNode video_output = fs["video_output"];
    if (!video_output.empty()) {
        video_output_fps = (float)(double)video_output["fps"];
        video_output_codec = (std::string)video_output["codec"];
    }
    
    // Camera extrinsics
    cv::FileNode extrinsics = fs["extrinsics"];
    if (!extrinsics.empty()) {
        cv::FileNode T_BC_node = extrinsics["T_BC"];
        if (!T_BC_node.empty()) {
            T_BC = LoadMatrix4f(T_BC_node);
        }
    }
    
    // IMU parameters
    cv::FileNode imu = fs["imu"];
    if (!imu.empty()) {
        imu_frequency = (float)(double)imu["frequency"];
        imu_gyro_noise = (float)(double)imu["gyro_noise"];
        imu_accel_noise = (float)(double)imu["accel_noise"];
        imu_gyro_bias_noise = (float)(double)imu["gyro_bias_noise"];
        imu_accel_bias_noise = (float)(double)imu["accel_bias_noise"];
    }
    
    fs.release();
    
    LOG_INFO("Config: {}x{}, {} features, {}x{} grid",
             camera_width, camera_height, max_features, grid_cols, grid_rows);
    
    return true;
}

Eigen::Matrix4f ConfigUtils::LoadMatrix4f(const cv::FileNode& node) {
    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
    
    if (node.isSeq() && node.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            cv::FileNode row = node[i];
            if (row.isSeq() && row.size() == 4) {
                for (int j = 0; j < 4; ++j) {
                    matrix(i, j) = (float)(double)row[j];
                }
            }
        }
    }
    
    return matrix;
}

} // namespace vio_360
