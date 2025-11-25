/**
 * @file      Initializer.h
 * @brief     Handles monocular and visual-inertial initialization
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
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
#include "../database/MapPoint.h"

namespace vio_360 {

/**
 * @brief Initialization result structure
 */
struct InitializationResult {
    bool success;
    Eigen::Matrix3f R;                    // Rotation from frame1 to frame2
    Eigen::Vector3f t;                    // Translation from frame1 to frame2 (scaled)
    std::vector<Eigen::Vector3f> points3d; // 3D points in world coordinates (scaled)
    std::vector<int> track_ids;           // Corresponding track IDs (feature IDs)
    int frame1_id;                        // First frame ID
    int frame2_id;                        // Second frame ID
    
    // Initialized keyframes and map points (like lightweight_vio)
    std::vector<std::shared_ptr<Frame>> initialized_keyframes;
    std::vector<std::shared_ptr<MapPoint>> initialized_mappoints;
    float scale_factor;                   // Scale factor used (1/median_depth)
    
    InitializationResult() 
        : success(false)
        , R(Eigen::Matrix3f::Identity())
        , t(Eigen::Vector3f::Zero())
        , frame1_id(-1)
        , frame2_id(-1)
        , scale_factor(1.0f) {}
};

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
     * @param result Output initialization result
     * @return True if initialization successful
     */
    bool TryMonocularInitialization(
        const std::vector<std::shared_ptr<Frame>>& frames,
        InitializationResult& result);
    
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
    
    /**
     * @brief Select features with sufficient observations for initialization
     * @param frames Frame window
     * @return Selected features
     */
    std::vector<std::shared_ptr<Feature>> SelectFeaturesForInit(
        const std::vector<std::shared_ptr<Frame>>& frames
    ) const;
    
    /**
     * @brief Interpolate poses for intermediate frames and register MapPoint observations
     * @param frames All frames in window (first and last should have poses set)
     * @param mappoints MapPoints created during initialization
     */
    void InterpolateIntermediateFrames(
        const std::vector<std::shared_ptr<Frame>>& frames,
        const std::vector<std::shared_ptr<MapPoint>>& mappoints
    );
    
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
    
    /**
     * @brief Triangulate a single 3D point using mid-point method
     * @param bearing1 Bearing vector from frame1
     * @param bearing2 Bearing vector from frame2
     * @param R Rotation from frame1 to frame2
     * @param t Translation from frame1 to frame2
     * @param point3d Output 3D point
     * @return True if triangulation successful
     */
    bool TriangulateSinglePoint(
        const Eigen::Vector3f& bearing1,
        const Eigen::Vector3f& bearing2,
        const Eigen::Matrix3f& R,
        const Eigen::Vector3f& t,
        Eigen::Vector3f& point3d
    ) const;
    
    /**
     * @brief Test a pose candidate using cheirality check
     * @param R Rotation candidate
     * @param t Translation candidate
     * @param bearings1 Bearing vectors from frame1
     * @param bearings2 Bearing vectors from frame2
     * @param inlier_mask Inlier mask from RANSAC
     * @return Number of points passing cheirality check
     */
    int TestPoseCandidate(
        const Eigen::Matrix3f& R,
        const Eigen::Vector3f& t,
        const std::vector<Eigen::Vector3f>& bearings1,
        const std::vector<Eigen::Vector3f>& bearings2,
        const std::vector<bool>& inlier_mask
    ) const;
    
    /**
     * @brief Compute reprojection error for a 3D point
     * @param point3d 3D point in frame1 coordinates
     * @param bearing_observed Observed bearing vector in frame2
     * @param R Rotation from frame1 to frame2
     * @param t Translation from frame1 to frame2
     * @return Angular error in radians
     */
    float ComputeReprojectionError(
        const Eigen::Vector3f& point3d,
        const Eigen::Vector3f& bearing_observed,
        const Eigen::Matrix3f& R,
        const Eigen::Vector3f& t
    ) const;
    
    /**
     * @brief Validate initialization quality
     * @param bearings1 Bearing vectors from frame1
     * @param bearings2 Bearing vectors from frame2
     * @param points3d Triangulated 3D points
     * @param R Rotation matrix
     * @param t Translation vector
     * @param inlier_mask Inlier mask
     * @param mean_error Output: mean reprojection error in pixels
     * @return True if validation passes
     */
    bool ValidateInitialization(
        const std::vector<Eigen::Vector3f>& bearings1,
        const std::vector<Eigen::Vector3f>& bearings2,
        const std::vector<Eigen::Vector3f>& points3d,
        const Eigen::Matrix3f& R,
        const Eigen::Vector3f& t,
        const std::vector<bool>& inlier_mask,
        float& mean_error
    ) const;
    
    /**
     * @brief Normalize scale so that median depth = 1.0
     * @param points3d 3D points to scale (modified in place)
     * @param t Translation vector to scale (modified in place)
     * @return Scale factor used (1/median_depth)
     */
    float NormalizeScale(
        std::vector<Eigen::Vector3f>& points3d,
        Eigen::Vector3f& t
    ) const;
    
    /**
     * @brief Create MapPoints and register to frames
     * @param frame1 First keyframe
     * @param frame2 Second keyframe
     * @param points3d Triangulated 3D points in world frame
     * @param selected_features Features used for triangulation
     * @param result Output: populated with mappoints
     */
    void CreateMapPoints(
        std::shared_ptr<Frame> frame1,
        std::shared_ptr<Frame> frame2,
        const std::vector<Eigen::Vector3f>& points3d,
        const std::vector<std::shared_ptr<Feature>>& selected_features,
        InitializationResult& result
    );
    
    /**
     * @brief Compute mean reprojection error for a frame using equirectangular projection
     * @param frame Frame to compute error for
     * @return Mean reprojection error in pixels
     */
    double ComputeFrameReprojectionError(const std::shared_ptr<Frame>& frame) const;

private:
    // ============ State ============
    bool m_is_initialized;
    
    // ============ Configuration (cached from ConfigUtils) ============
    int m_camera_width;              // Camera image width
    int m_camera_height;             // Camera image height
    int m_min_features;              // Minimum features for initialization
    int m_min_observations;          // Minimum observations per feature
    float m_min_parallax;            // Minimum parallax in pixels
    float m_ransac_threshold;        // RANSAC threshold for inlier
    int m_ransac_iterations;         // RANSAC iterations
    float m_min_inlier_ratio;        // Minimum inlier ratio for success
    float m_max_reprojection_error;  // Maximum reprojection error
    int m_init_grid_cols;            // Grid columns for feature sampling
    int m_init_grid_rows;            // Grid rows for feature sampling
};

} // namespace vio_360
