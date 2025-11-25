/**
 * @file      Initializer.h
 * @brief     Handles monocular and visual-inertial initialization
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "../database/Frame.h"
#include "../database/Feature.h"

namespace vio_360 {

/**
 * @brief Initializer class for monocular and visual-inertial initialization
 * 
 * Handles:
 * 1. Monocular initialization: 2-view geometry, triangulation
 * 2. Visual-Inertial initialization: IMU pre-integration, alignment (future)
 */
class Initializer {
public:
    Initializer();
    ~Initializer() = default;

    // ============ Monocular Initialization ============
    
    /**
     * @brief Attempt monocular initialization with current frames
     * @param frames Window of frames to use for initialization
     * @return True if initialization successful
     */
    bool TryMonocularInitialization(const std::vector<std::shared_ptr<Frame>>& frames);
    
    /**
     * @brief Compute median parallax between two frames using tracked features
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Median parallax in pixels
     */
    float ComputeParallax(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2
    ) const;
    
    // ============ Status & Results ============
    
    /**
     * @brief Check if initialization is complete
     */
    bool IsInitialized() const { return m_is_initialized; }
    
    /**
     * @brief Reset initialization state
     */
    void Reset();
    
    // ============ Future: IMU Initialization ============
    // bool TryVisualInertialInitialization(...);
    // void EstimateGyroBias(...);
    // void EstimateGravityDirection(...);

private:
    // ============ Monocular Initialization Helpers ============
    
    /**
     * @brief Select features with sufficient observations for initialization
     * @param frames Frame window
     * @return Selected features
     */
    std::vector<std::shared_ptr<Feature>> SelectFeaturesForInit(
        const std::vector<std::shared_ptr<Frame>>& frames
    ) const;
    
    /**
     * @brief Select best frame pair based on parallax
     * @param features Features with observations
     * @param frame1 Output: first frame
     * @param frame2 Output: second frame
     * @return True if valid pair found
     */
    bool SelectFramePair(
        const std::vector<std::shared_ptr<Feature>>& features,
        std::shared_ptr<Frame>& frame1,
        std::shared_ptr<Frame>& frame2
    ) const;
    
    /**
     * @brief Compute Essential matrix using RANSAC
     * @param bearings1 Bearing vectors from frame1
     * @param bearings2 Bearing vectors from frame2
     * @param E Output Essential matrix
     * @param inlier_mask Output inlier mask
     * @return True if Essential matrix found
     */
    bool ComputeEssentialMatrix(
        const std::vector<Eigen::Vector3f>& bearings1,
        const std::vector<Eigen::Vector3f>& bearings2,
        Eigen::Matrix3f& E,
        std::vector<bool>& inlier_mask
    ) const;
    
    /**
     * @brief Recover pose (R, t) from Essential matrix
     * @param E Essential matrix
     * @param bearings1 Bearing vectors from frame1
     * @param bearings2 Bearing vectors from frame2
     * @param R Output rotation
     * @param t Output translation (unit scale)
     * @param inlier_mask Inlier mask from RANSAC
     * @return True if pose recovered successfully
     */
    bool RecoverPose(
        const Eigen::Matrix3f& E,
        const std::vector<Eigen::Vector3f>& bearings1,
        const std::vector<Eigen::Vector3f>& bearings2,
        Eigen::Matrix3f& R,
        Eigen::Vector3f& t,
        const std::vector<bool>& inlier_mask
    ) const;
    
    /**
     * @brief Triangulate 3D points from two views
     * @param bearings1 Bearing vectors from frame1
     * @param bearings2 Bearing vectors from frame2
     * @param R Rotation from frame1 to frame2
     * @param t Translation from frame1 to frame2
     * @param points3d Output 3D points
     * @return Number of successfully triangulated points
     */
    int TriangulatePoints(
        const std::vector<Eigen::Vector3f>& bearings1,
        const std::vector<Eigen::Vector3f>& bearings2,
        const Eigen::Matrix3f& R,
        const Eigen::Vector3f& t,
        std::vector<Eigen::Vector3f>& points3d
    ) const;

private:
    // ============ State ============
    bool m_is_initialized;
    
    // ============ Configuration ============
    int m_min_features;              // Minimum features for initialization
    int m_min_observations;          // Minimum observations per feature
    float m_min_parallax;            // Minimum parallax in pixels
    float m_ransac_threshold;        // RANSAC threshold for inlier
    int m_ransac_iterations;         // RANSAC iterations
    float m_min_inlier_ratio;        // Minimum inlier ratio for success
    float m_max_reprojection_error;  // Maximum reprojection error
};

} // namespace vio_360
