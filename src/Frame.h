/**
 * @file      Frame.h
 * @brief     Frame class for 360-degree VIO
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "Feature.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>

namespace vio_360 {

/**
 * @brief IMU measurement structure
 */
struct IMUData {
    double timestamp;              // timestamp in seconds
    Eigen::Vector3f linear_accel;  // linear acceleration [m/s^2]
    Eigen::Vector3f angular_vel;   // angular velocity [rad/s]
    
    IMUData() = default;
    IMUData(double ts, const Eigen::Vector3f& accel, const Eigen::Vector3f& gyro)
        : timestamp(ts), linear_accel(accel), angular_vel(gyro) {}
};

/**
 * @brief Frame class representing a single 360-degree image capture
 */
class Frame {
public:
    /**
     * @brief Constructor
     * @param timestamp Frame timestamp in nanoseconds
     * @param frame_id Unique frame identifier
     * @param image Grayscale image (already resized if needed)
     * @param width Original image width
     * @param height Original image height
     */
    Frame(long long timestamp, int frame_id, const cv::Mat& image, int width, int height);
    
    ~Frame() = default;
    
    // ============ Getters ============
    
    long long GetTimestamp() const { return m_timestamp; }
    int GetFrameId() const { return m_frame_id; }
    const cv::Mat& GetImage() const { return m_image; }
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    
    const std::vector<std::shared_ptr<Feature>>& GetFeatures() const { return m_features; }
    std::vector<std::shared_ptr<Feature>>& GetFeaturesMutable() { return m_features; }
    size_t GetFeatureCount() const { return m_features.size(); }
    
    bool IsKeyframe() const { return m_is_keyframe; }
    
    // ============ Pose Management ============
    
    /**
     * @brief Get body pose in world frame (T_wb)
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4f GetTwb() const;
    
    /**
     * @brief Set body pose in world frame (T_wb)
     * @param T_wb 4x4 transformation matrix
     */
    void SetTwb(const Eigen::Matrix4f& T_wb);
    
    /**
     * @brief Get camera pose in world frame (T_wc)
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4f GetTwc() const;
    
    /**
     * @brief Get rotation matrix (world to body)
     */
    Eigen::Matrix3f GetRotation() const { return m_rotation; }
    
    /**
     * @brief Get translation vector (world to body)
     */
    Eigen::Vector3f GetTranslation() const { return m_translation; }
    
    // ============ VIO State ============
    
    Eigen::Vector3f GetVelocity() const { return m_velocity; }
    void SetVelocity(const Eigen::Vector3f& velocity) { m_velocity = velocity; }
    
    Eigen::Vector3f GetAccelBias() const { return m_accel_bias; }
    void SetAccelBias(const Eigen::Vector3f& bias) { m_accel_bias = bias; }
    
    Eigen::Vector3f GetGyroBias() const { return m_gyro_bias; }
    void SetGyroBias(const Eigen::Vector3f& bias) { m_gyro_bias = bias; }
    
    // ============ Keyframe Management ============
    
    void SetKeyframe(bool is_keyframe) { m_is_keyframe = is_keyframe; }
    
    double GetDtFromLastKeyframe() const { return m_dt_from_last_keyframe; }
    void SetDtFromLastKeyframe(double dt) { m_dt_from_last_keyframe = dt; }
    
    // ============ IMU Data Management ============
    
    const std::vector<IMUData>& GetIMUDataFromLastFrame() const { return m_imu_data_from_last_frame; }
    void SetIMUDataFromLastFrame(const std::vector<IMUData>& imu_data) { m_imu_data_from_last_frame = imu_data; }
    bool HasIMUData() const { return !m_imu_data_from_last_frame.empty(); }
    
    // ============ Feature Management ============
    
    void AddFeature(std::shared_ptr<Feature> feature) { m_features.push_back(feature); }
    void ClearFeatures() { m_features.clear(); }
    
    // ============ Grid-based Feature Management ============
    
    /**
     * @brief Set grid parameters for feature distribution
     * @param grid_cols Number of grid columns
     * @param grid_rows Number of grid rows
     * @param max_per_grid Maximum features per grid cell
     */
    void SetGridParameters(int grid_cols, int grid_rows, int max_per_grid);
    
    /**
     * @brief Assign features to grid cells based on pixel coordinates
     */
    void AssignFeaturesToGrid();
    
    /**
     * @brief Limit features per grid cell (keep strongest features)
     * Marks weak features as invalid based on track count
     */
    void LimitFeaturesPerGrid();
    
    /**
     * @brief Select best features from each grid cell for tracking
     * @return Vector of feature indices to track
     */
    std::vector<int> SelectFeaturesForTracking() const;
    
    /**
     * @brief Get grid parameters
     */
    int GetGridCols() const { return m_grid_cols; }
    int GetGridRows() const { return m_grid_rows; }
    int GetMaxFeaturesPerGrid() const { return m_max_features_per_grid; }
    
    // ============ Camera Extrinsics ============
    
    /**
     * @brief Set camera to body transformation (T_BC)
     * @param T_BC 4x4 transformation matrix from camera to body
     */
    void SetTBC(const Eigen::Matrix4f& T_BC);
    
    /**
     * @brief Get camera to body transformation (T_BC)
     */
    Eigen::Matrix4f GetTBC() const { return m_T_BC; }
    
    /**
     * @brief Get body to camera transformation (T_CB)
     */
    Eigen::Matrix4f GetTCB() const { return m_T_CB; }
    
private:
    // Frame information
    long long m_timestamp;         // Timestamp in nanoseconds
    int m_frame_id;                // Unique frame ID
    cv::Mat m_image;               // Grayscale image
    int m_width;                   // Image width
    int m_height;                  // Image height
    
    // Features
    std::vector<std::shared_ptr<Feature>> m_features;
    
    // Pose (body frame in world frame)
    Eigen::Matrix3f m_rotation;    // Rotation matrix R_wb
    Eigen::Vector3f m_translation; // Translation vector t_wb
    mutable std::mutex m_pose_mutex;
    
    // VIO state
    Eigen::Vector3f m_velocity;    // Velocity in world frame [m/s]
    Eigen::Vector3f m_accel_bias;  // Accelerometer bias [m/s^2]
    Eigen::Vector3f m_gyro_bias;   // Gyroscope bias [rad/s]
    
    // Keyframe flag
    bool m_is_keyframe;
    double m_dt_from_last_keyframe; // Time since last keyframe [seconds]
    
    // IMU data
    std::vector<IMUData> m_imu_data_from_last_frame;
    
    // Camera extrinsics (body to camera transformation)
    Eigen::Matrix4f m_T_BC;        // Camera to body
    Eigen::Matrix4f m_T_CB;        // Body to camera
    
    // Grid parameters for feature distribution
    int m_grid_cols;               // Number of grid columns (default: 20)
    int m_grid_rows;               // Number of grid rows (default: 10)
    int m_max_features_per_grid;   // Max features per cell (default: 4)
    
    // Grid structure: [row][col] = vector of feature indices
    std::vector<std::vector<std::vector<int>>> m_feature_grid;
};

} // namespace vio_360
