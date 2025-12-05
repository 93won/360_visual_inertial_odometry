/**
 * @file      Optimizer.h
 * @brief     Defines PnP and Bundle Adjustment optimizers using Ceres Solver
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
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/core.hpp>

namespace vio_360 {

// Forward declarations
class Frame;
class MapPoint;
class Camera;

/**
 * @brief PnP optimization result
 */
struct PnPResult {
    bool success;
    int num_inliers;
    int num_outliers;
    Eigen::Matrix4f optimized_pose;
    double initial_cost;
    double final_cost;
    int num_iterations;
    
    PnPResult() : success(false), num_inliers(0), num_outliers(0),
                  optimized_pose(Eigen::Matrix4f::Identity()),
                  initial_cost(0.0), final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Bundle Adjustment result
 */
struct BAResult {
    bool success;
    int num_inliers;
    int num_outliers;
    int num_poses_optimized;
    int num_points_optimized;
    double initial_cost;
    double final_cost;
    int num_iterations;
    
    BAResult() : success(false), num_inliers(0), num_outliers(0),
                 num_poses_optimized(0), num_points_optimized(0),
                 initial_cost(0.0), final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Optimizer class for PnP and Bundle Adjustment
 */
class Optimizer {
public:
    /**
     * @brief Constructor
     */
    Optimizer();
    
    /**
     * @brief Destructor
     */
    ~Optimizer() = default;
    
    /**
     * @brief Set camera for boundary checking
     * @param camera Camera pointer
     * @param boundary_margin Boundary margin in pixels
     */
    void SetCamera(std::shared_ptr<Camera> camera, int boundary_margin = 20);
    
    /**
     * @brief Solve PnP for a single frame using observed MapPoints
     * @param frame Frame to optimize pose for (must have MapPoint observations)
     * @param fix_mappoints If true, MapPoints are not modified
     * @return PnP optimization result
     */
    PnPResult SolvePnP(std::shared_ptr<Frame> frame, bool fix_mappoints = true);
    
    /**
     * @brief Run Bundle Adjustment on a set of frames and their MapPoints
     * @param frames Vector of frames to optimize
     * @param fix_first_pose If true, first frame's pose is fixed
     * @param fix_last_pose If true, last frame's pose is fixed
     * @return BA optimization result
     */
    BAResult RunBA(const std::vector<std::shared_ptr<Frame>>& frames,
                   bool fix_first_pose = true,
                   bool fix_last_pose = false);
    
    /**
     * @brief Run Local BA with sliding window
     * 
     * Optimizes poses of window frames and all MapPoints observed by them.
     * MapPoints that are observed by frames outside the window will include
     * those observations as constraints with FIXED poses.
     * 
     * @param window_frames Vector of frames in the sliding window
     *        - First frame is always fixed (gauge freedom)
     *        - Other window frames are optimized
     *        - Frames outside window that observe the MapPoints are fixed
     * @return BA optimization result
     */
    BAResult RunLocalBA(const std::vector<std::shared_ptr<Frame>>& window_frames);
    
    /**
     * @brief Run Full BA on initialization window
     * @param frames Vector of frames (first and last have known poses from Essential)
     * @return BA optimization result
     */
    BAResult RunFullBA(const std::vector<std::shared_ptr<Frame>>& frames);

private:
    /**
     * @brief Convert frame pose to parameter array [rot(3), trans(3)]
     */
    void PoseToParams(const Eigen::Matrix4f& pose, double* params);
    
    /**
     * @brief Convert parameter array to pose matrix
     */
    Eigen::Matrix4f ParamsToPose(const double* params);
    
    /**
     * @brief Setup Ceres solver options
     */
    ceres::Solver::Options SetupSolverOptions(int max_iterations = 50);
    
    /**
     * @brief Check if a feature is near horizontal boundary
     */
    bool IsNearBoundary(const cv::Point2f& pixel) const;
    
    // Parameters
    double m_huber_delta;         // Huber loss delta
    double m_pixel_noise_std;     // Pixel noise standard deviation
    int m_max_iterations;         // Maximum iterations
    double m_chi2_threshold;      // Chi-square threshold for outlier detection
    
    // Camera for boundary checking
    std::shared_ptr<Camera> m_camera;
    int m_boundary_margin;        // Boundary margin in pixels
};

} // namespace vio_360
