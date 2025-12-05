/**
 * @file      Estimator.h
 * @brief     Main VIO estimator for 360 ERP images
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace vio_360 {

// Forward declarations
class Frame;
class Camera;
class FeatureTracker;
class Initializer;
class IMUPreintegrator;

/**
 * @brief IMU measurement data structure
 */
struct IMUData {
    double timestamp;    // seconds
    float ax, ay, az;    // m/s^2
    float gx, gy, gz;    // rad/s
};

/**
 * @brief Main VIO estimator class
 */
class Estimator {
public:
    /**
     * @brief VIO estimation result
     */
    struct EstimationResult {
        bool success;
        Eigen::Matrix4f pose;          // Current pose (T_wc: world to camera)
        int num_features;              // Total features in current frame
        int num_tracked;               // Successfully tracked features
        int num_inliers;               // Inliers after RANSAC
        bool init_ready;               // Initialization is ready (sufficient parallax)
        bool init_success;             // Initialization successfully completed
        
        EstimationResult() 
            : success(false)
            , pose(Eigen::Matrix4f::Identity())
            , num_features(0)
            , num_tracked(0)
            , num_inliers(0)
            , init_ready(false)
            , init_success(false) {}
    };

    /**
     * @brief Constructor
     */
    Estimator();
    
    /**
     * @brief Destructor
     */
    ~Estimator();

    /**
     * @brief Process a new monocular frame
     * @param image Input ERP image
     * @param timestamp Frame timestamp in seconds
     * @return Estimation result
     */
    EstimationResult ProcessFrame(const cv::Mat& image, double timestamp);
    
    /**
     * @brief Process a new monocular frame with IMU data (VIO mode)
     * @param image Input ERP image
     * @param timestamp Frame timestamp in seconds
     * @param imu_data IMU measurements between previous and current frame
     * @return Estimation result
     */
    EstimationResult ProcessFrame(const cv::Mat& image, double timestamp, 
                                  const std::vector<IMUData>& imu_data);

    /**
     * @brief Try to initialize monocular VO
     * @return True if initialization successful
     */
    bool TryInitialize();

    /**
     * @brief Reset the estimator state
     */
    void Reset();

    /**
     * @brief Get current pose
     * @return Current camera pose (T_wc)
     */
    Eigen::Matrix4f GetCurrentPose() const { return m_current_pose; }

    /**
     * @brief Get current frame
     * @return Current frame (can be nullptr)
     */
    std::shared_ptr<Frame> GetCurrentFrame() const { return m_current_frame; }

    /**
     * @brief Get all frames
     * @return Vector of all processed frames
     */
    const std::vector<std::shared_ptr<Frame>>& GetAllFrames() const { return m_all_frames; }

    /**
     * @brief Check if system is initialized
     * @return True if initialized
     */
    bool IsInitialized() const { return m_initialized; }
    
    /**
     * @brief Get initialized 3D points (map points from initialization)
     * @return Vector of 3D points in world coordinates
     */
    const std::vector<Eigen::Vector3f>& GetInitializedPoints() const { return m_initialized_points; }
    
    /**
     * @brief Get initialization camera poses (frame1 and frame2)
     * @return Vector of 4x4 transformation matrices [T_w1, T_w2]
     */
    const std::vector<Eigen::Matrix4f>& GetInitializationPoses() const { return m_init_poses; }
    
    /**
     * @brief Get frame window (keyframes used in initialization)
     * @return Vector of frames in the current window
     */
    const std::vector<std::shared_ptr<Frame>>& GetFrameWindow() const { return m_frame_window; }
    
    /**
     * @brief Get all keyframes
     * @return Vector of all keyframes
     */
    const std::vector<std::shared_ptr<Frame>>& GetKeyframes() const { return m_keyframes; }
    
    /**
     * @brief Get all keyframes including those outside the window
     * @return Vector of all keyframes ever created
     */
    const std::vector<std::shared_ptr<Frame>>& GetAllKeyframes() const { return m_all_keyframes; }
    
    /**
     * @brief Check if a keyframe is in the current sliding window
     * @param frame_id Frame ID to check
     * @return True if the keyframe is in the window
     */
    bool IsKeyframeInWindow(int frame_id) const;
    
    /**
     * @brief Get the number of keyframes in the sliding window
     * @return Number of keyframes in window
     */
    int GetWindowSize() const { return static_cast<int>(m_keyframes.size()); }

private:
    // System components
    std::unique_ptr<FeatureTracker> m_feature_tracker;
    std::unique_ptr<Initializer> m_initializer;
    std::unique_ptr<IMUPreintegrator> m_imu_preintegrator;
    std::shared_ptr<Camera> m_camera;
    
    // State
    std::shared_ptr<Frame> m_current_frame;
    std::shared_ptr<Frame> m_previous_frame;
    std::shared_ptr<Frame> m_last_keyframe;
    std::vector<std::shared_ptr<Frame>> m_all_frames;
    std::vector<std::shared_ptr<Frame>> m_keyframes;         // Sliding window keyframes
    std::vector<std::shared_ptr<Frame>> m_all_keyframes;     // All keyframes (including out of window)
    
    // Frame window for initialization
    std::vector<std::shared_ptr<Frame>> m_frame_window;
    int m_window_size;
    
    // Frame management
    int m_frame_id_counter;
    
    // Initialization state
    bool m_initialized;
    float m_min_parallax;
    
    // Initialized map points and poses
    std::vector<Eigen::Vector3f> m_initialized_points;  // 3D points from initialization
    std::vector<Eigen::Matrix4f> m_init_poses;          // [T_w1, T_w2] poses from initialization
    
    // Current pose (T_wc: world to camera)
    Eigen::Matrix4f m_current_pose;
    
    // Constant velocity model: transformation from last frame to current frame
    Eigen::Matrix4f m_transform_from_last;
    
    /**
     * @brief Create a new frame
     * @param image Input image
     * @param timestamp Frame timestamp
     * @return New frame
     */
    std::shared_ptr<Frame> CreateFrame(const cv::Mat& image, double timestamp);
    
    /**
     * @brief Track features from previous frame
     * @return Number of tracked features
     */
    int TrackFeatures();
    
    /**
     * @brief Detect new features in current frame
     * @return Number of detected features
     */
    int DetectFeatures();
    
    /**
     * @brief Check if current frame should be a keyframe
     * @return True if should create keyframe
     */
    bool ShouldCreateKeyframe();
    
    /**
     * @brief Create a new keyframe from current frame
     */
    void CreateKeyframe();
    
    /**
     * @brief Process IMU data and compute preintegration
     * @param imu_data IMU measurements between frames
     */
    void ProcessIMU(const std::vector<IMUData>& imu_data);
    
    /**
     * @brief Link MapPoints from previous frame to current frame based on feature tracking
     */
    void LinkMapPointsFromPreviousFrame();
    
    /**
     * @brief Process intermediate frames after initialization
     * Interpolate poses, link MapPoints, run PnP for frames between keyframes
     */
    void ProcessIntermediateFrames();
    
    /**
     * @brief Compute median parallax between two frames
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Median parallax in pixels
     */
    float ComputeParallax(const std::shared_ptr<Frame>& frame1,
                          const std::shared_ptr<Frame>& frame2) const;
    
    /**
     * @brief Triangulate a single 3D point from two bearing vectors
     * @param bearing1 Bearing vector in frame1
     * @param bearing2 Bearing vector in frame2
     * @param T1w Transform from world to frame1
     * @param T2w Transform from world to frame2
     * @param point3d Output 3D point in world frame
     * @return True if triangulation succeeded
     */
    bool TriangulateSinglePoint(
        const Eigen::Vector3f& bearing1,
        const Eigen::Vector3f& bearing2,
        const Eigen::Matrix4f& T1w,
        const Eigen::Matrix4f& T2w,
        Eigen::Vector3f& point3d) const;
    
    /**
     * @brief Triangulate new MapPoints between two keyframes
     * @param kf1 First keyframe (reference)
     * @param kf2 Second keyframe (current)
     * @return Number of successfully triangulated MapPoints
     */
    int TriangulateNewMapPoints(
        const std::shared_ptr<Frame>& kf1,
        const std::shared_ptr<Frame>& kf2);
    
    /**
     * @brief Accumulate IMU data since last keyframe
     */
    std::vector<IMUData> m_imu_since_last_keyframe;
};

} // namespace vio_360
