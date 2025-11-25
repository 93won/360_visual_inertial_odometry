/**
 * @file      FeatureTracker.h
 * @brief     Feature tracking on Equirectangular (360-degree) images.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace vio_360 {

// Forward declarations
class Camera;
class Feature;
class Frame;

/**
 * @brief Feature tracker for 360-degree ERP images
 */
class FeatureTracker {
public:
    /**
     * @brief Constructor with camera and parameters
     * @param camera Shared pointer to Camera object
     * @param max_features Maximum number of features to track
     * @param min_distance Minimum distance between features (pixels)
     * @param quality_level Quality threshold for feature detection
     */
    FeatureTracker(std::shared_ptr<Camera> camera,
                   int max_features = 500,
                   float min_distance = 30.0f,
                   float quality_level = 0.01f);
    
    ~FeatureTracker() = default;

    // ============ Main Tracking Interface ============
    
    /**
     * @brief Track features in current frame
     * @param current_frame Current frame with image
     * @param previous_frame Previous frame (optional, can be nullptr)
     */
    void TrackFeatures(std::shared_ptr<Frame> current_frame,
                      std::shared_ptr<Frame> previous_frame = nullptr);
    
    // ============ Feature Detection ============
    
    /**
     * @brief Detect new features in the image
     * @param image Grayscale image
     * @param mask Mask image (optional)
     * @return Vector of detected feature points
     */
    std::vector<cv::Point2f> DetectNewFeatures(const cv::Mat& image, 
                                                const cv::Mat& mask = cv::Mat());
    
    // ============ Optical Flow Tracking ============
    
    /**
     * @brief Track features using Lucas-Kanade optical flow
     * @param prev_image Previous grayscale image
     * @param curr_image Current grayscale image
     * @param prev_points Previous feature points
     * @param curr_points Output current feature points
     * @param status Output status for each point
     * @return Number of successfully tracked points
     */
    int TrackOpticalFlow(const cv::Mat& prev_image,
                        const cv::Mat& curr_image,
                        const std::vector<cv::Point2f>& prev_points,
                        std::vector<cv::Point2f>& curr_points,
                        std::vector<uchar>& status);
    
    // ============ Outlier Rejection ============
    
    /**
     * @brief Reject outliers using rotation-only RANSAC
     * @param prev_points Previous feature points
     * @param curr_points Current feature points
     * @param inlier_mask Output inlier mask
     * @return Number of inliers
     */
    int RejectOutliersRotationRANSAC(const std::vector<cv::Point2f>& prev_points,
                                     const std::vector<cv::Point2f>& curr_points,
                                     std::vector<uchar>& inlier_mask);
    
    // ============ Grid-based Feature Management ============
    
    /**
     * @brief Manage features using grid-based distribution
     * @param features Vector of features to manage
     */
    void ManageGridFeatures(std::vector<std::shared_ptr<Feature>>& features);
    
    /**
     * @brief Create mask for new feature detection
     * @param existing_features Existing features to avoid
     * @return Mask image
     */
    cv::Mat CreateFeatureMask(const std::vector<std::shared_ptr<Feature>>& existing_features);
    
    /**
     * @brief Remove clustered features from frame
     * @param frame Frame to process
     */
    void RemoveClusteredFeatures(std::shared_ptr<Frame> frame);
    
    // ============ Utility Functions ============
    
    /**
     * @brief Update bearing vectors for all features
     * @param features Vector of features
     */
    void UpdateBearingVectors(std::vector<std::shared_ptr<Feature>>& features);
    
    /**
     * @brief Get statistics about current tracking
     * @param num_tracked Output number of tracked features
     * @param num_detected Output number of newly detected features
     */
    void GetTrackingStats(int& num_tracked, int& num_detected) const;
    
    // ============ Getters/Setters ============
    
    void SetMaxFeatures(int max_features) { m_max_features = max_features; }
    void SetMinDistance(float min_distance) { m_min_distance = min_distance; }
    int GetMaxFeatures() const { return m_max_features; }
    
private:
    // ============ Helper Functions ============
    
    /**
     * @brief Estimate rotation matrix from bearing correspondences
     * @param bearings1 First set of bearing vectors (Nx3)
     * @param bearings2 Second set of bearing vectors (Nx3)
     * @return 3x3 rotation matrix
     */
    Eigen::Matrix3f EstimateRotation(const std::vector<Eigen::Vector3f>& bearings1,
                                     const std::vector<Eigen::Vector3f>& bearings2);
    
    /**
     * @brief Compute inliers for rotation model
     * @param bearings1 First set of bearing vectors
     * @param bearings2 Second set of bearing vectors
     * @param rotation Rotation matrix
     * @param threshold Angular threshold in radians
     * @return Inlier mask
     */
    std::vector<uchar> ComputeRotationInliers(
        const std::vector<Eigen::Vector3f>& bearings1,
        const std::vector<Eigen::Vector3f>& bearings2,
        const Eigen::Matrix3f& rotation,
        float threshold);
    
    // ============ Member Variables ============
    
    std::shared_ptr<Camera> m_camera;        // Camera model
    
    // Feature detection parameters
    int m_max_features;                      // Maximum number of features
    float m_min_distance;                    // Minimum distance between features
    float m_quality_level;                   // Quality threshold for corners
    
    // Optical flow parameters
    cv::Size m_lk_window_size;               // LK window size
    int m_lk_max_level;                      // Pyramid levels
    cv::TermCriteria m_lk_criteria;          // Termination criteria
    
    // RANSAC parameters
    int m_ransac_max_iterations;             // Maximum RANSAC iterations
    float m_ransac_threshold;                // Angular threshold (degrees)
    float m_ransac_confidence;               // RANSAC confidence
    
    // Grid parameters
    int m_grid_cols;                         // Number of grid columns
    int m_grid_rows;                         // Number of grid rows
    int m_max_features_per_grid;             // Max features per grid cell
    
    // Tracking state
    cv::Mat m_prev_image;                    // Previous image
    std::vector<std::shared_ptr<Feature>> m_prev_features;  // Previous features
    int m_next_feature_id;                   // Next available feature ID
    
    // Statistics
    int m_num_tracked;                       // Number of tracked features
    int m_num_detected;                      // Number of newly detected features
    
    // Mask
    cv::Mat m_polar_mask;                    // Polar region mask
};

} // namespace vio_360
