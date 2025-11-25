/**
 * @file      Estimator.h
 * @brief     Main VIO estimator for 360 ERP images
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
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
        
        EstimationResult() 
            : success(false)
            , pose(Eigen::Matrix4f::Identity())
            , num_features(0)
            , num_tracked(0)
            , num_inliers(0) {}
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

private:
    // System components
    std::unique_ptr<FeatureTracker> m_feature_tracker;
    std::unique_ptr<Initializer> m_initializer;
    std::shared_ptr<Camera> m_camera;
    
    // State
    std::shared_ptr<Frame> m_current_frame;
    std::shared_ptr<Frame> m_previous_frame;
    std::shared_ptr<Frame> m_last_keyframe;
    std::vector<std::shared_ptr<Frame>> m_all_frames;
    std::vector<std::shared_ptr<Frame>> m_keyframes;
    
    // Frame window for initialization
    std::vector<std::shared_ptr<Frame>> m_frame_window;
    int m_window_size;
    
    // Frame management
    int m_frame_id_counter;
    
    // Initialization state
    bool m_initialized;
    float m_min_parallax;
    
    // Current pose (T_wc: world to camera)
    Eigen::Matrix4f m_current_pose;
    
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
};

} // namespace vio_360
