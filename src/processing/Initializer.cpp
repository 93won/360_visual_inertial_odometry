/**
 * @file      Initializer.cpp
 * @brief     Implementation of Initializer
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Initializer.h"
#include "ConfigUtils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>

namespace vio_360 {

Initializer::Initializer()
    : m_is_initialized(false)
{
    // Load configuration
    const auto& config = ConfigUtils::GetInstance();
    
    m_camera_width = config.camera_width;
    m_camera_height = config.camera_height;
    m_min_features = config.initialization_min_features;
    m_min_observations = config.initialization_min_observations;
    m_min_parallax = config.initialization_min_parallax;
    m_ransac_threshold = config.initialization_ransac_threshold;
    m_ransac_iterations = config.initialization_ransac_iterations;
    m_min_inlier_ratio = config.initialization_min_inlier_ratio;
    m_max_reprojection_error = config.initialization_max_reprojection_error;
    m_init_grid_cols = config.initialization_grid_cols;
    m_init_grid_rows = config.initialization_grid_rows;
    
    std::cout << "\n[Initializer] Configuration loaded:" << std::endl;
    std::cout << "  Camera: " << m_camera_width << "x" << m_camera_height << std::endl;
    std::cout << "  Min features: " << m_min_features << std::endl;
    std::cout << "  Min observations: " << m_min_observations << std::endl;
    std::cout << "  Min parallax: " << m_min_parallax << " pixels" << std::endl;
    std::cout << "  RANSAC threshold: " << m_ransac_threshold << " radians" << std::endl;
    std::cout << "  RANSAC iterations: " << m_ransac_iterations << std::endl;
    std::cout << "  Min inlier ratio: " << m_min_inlier_ratio << std::endl;
    std::cout << "  Max reprojection error: " << m_max_reprojection_error << " pixels" << std::endl;
    std::cout << "  Grid: " << m_init_grid_cols << "x" << m_init_grid_rows << std::endl;
}

void Initializer::Reset() {
    m_is_initialized = false;
}

bool Initializer::TryMonocularInitialization(
    const std::vector<std::shared_ptr<Frame>>& frames,
    InitializationResult& result
) {
    std::cout << "\n========== Starting Monocular Initialization ==========" << std::endl;
    
    // Reset result
    result = InitializationResult();
    
    // 1. Select features with sufficient observations
    auto selected_features = SelectFeaturesForInit(frames);
    
    if (selected_features.size() < static_cast<size_t>(m_min_features)) {
        std::cout << "[Initializer] Not enough features selected" << std::endl;
        return false;
    }
    
    // 2. Select frame pair
    std::shared_ptr<Frame> frame1, frame2;
    if (!SelectFramePair(selected_features, frame1, frame2)) {
        std::cout << "[Initializer] Failed to select frame pair" << std::endl;
        return false;
    }
    
    // 3. Extract bearing vectors for corresponding features
    std::vector<Eigen::Vector3f> bearings1, bearings2;
    bearings1.reserve(selected_features.size());
    bearings2.reserve(selected_features.size());
    
    for (const auto& feat : selected_features) {
        // Get observations from this feature
        const auto& observations = feat->GetObservations();
        
        // Find observations in frame1 and frame2
        std::shared_ptr<Frame> obs_frame1 = nullptr;
        std::shared_ptr<Frame> obs_frame2 = nullptr;
        int feat_idx1 = -1, feat_idx2 = -1;
        
        for (const auto& obs : observations) {
            if (obs.frame->GetFrameId() == frame1->GetFrameId()) {
                obs_frame1 = obs.frame;
                feat_idx1 = obs.feature_index;
            }
            if (obs.frame->GetFrameId() == frame2->GetFrameId()) {
                obs_frame2 = obs.frame;
                feat_idx2 = obs.feature_index;
            }
        }
        
        if (!obs_frame1 || !obs_frame2 || feat_idx1 < 0 || feat_idx2 < 0) {
            continue; // Skip if not observed in both frames
        }
        
        // Get bearing vectors
        const auto& features1 = obs_frame1->GetFeatures();
        const auto& features2 = obs_frame2->GetFeatures();
        
        if (feat_idx1 >= static_cast<int>(features1.size()) || 
            feat_idx2 >= static_cast<int>(features2.size())) {
            continue;
        }
        
        bearings1.push_back(features1[feat_idx1]->GetBearing());
        bearings2.push_back(features2[feat_idx2]->GetBearing());
    }
    
    std::cout << "\n[Initializer] Extracted " << bearings1.size() 
              << " bearing correspondences" << std::endl;
    
    if (bearings1.size() < 5) {
        std::cout << "[Initializer] Not enough bearing correspondences" << std::endl;
        return false;
    }
    
    // 4. Compute Essential matrix
    Eigen::Matrix3f E;
    std::vector<bool> inlier_mask;
    
    if (!ComputeEssentialMatrix(bearings1, bearings2, E, inlier_mask)) {
        std::cout << "[Initializer] Failed to compute Essential matrix" << std::endl;
        return false;
    }
    
    // 5. Recover pose (R, t) from Essential matrix
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    
    if (!RecoverPose(E, bearings1, bearings2, R, t, inlier_mask)) {
        std::cout << "[Initializer] Failed to recover pose" << std::endl;
        return false;
    }
    
    // 6. Triangulate all inlier points
    std::vector<Eigen::Vector3f> points3d;
    int num_triangulated = TriangulatePoints(bearings1, bearings2, R, t, points3d);
    
    if (num_triangulated < m_min_features) {
        std::cout << "[Initializer] Not enough triangulated points (got " 
                  << num_triangulated << ", need " << m_min_features << ")" << std::endl;
        return false;
    }
    
    // 7. Validate initialization quality
    if (!ValidateInitialization(bearings1, bearings2, points3d, R, t, inlier_mask)) {
        std::cout << "[Initializer] Validation failed" << std::endl;
        return false;
    }
    
    // 8. Success! Mark as initialized
    m_is_initialized = true;
    
    // 9. Populate result structure
    result.success = true;
    result.R = R;
    result.t = t;
    result.points3d = points3d;
    result.frame1_id = frame1->GetFrameId();
    result.frame2_id = frame2->GetFrameId();
    
    // Collect track IDs for the triangulated points
    result.track_ids.clear();
    result.track_ids.reserve(selected_features.size());
    for (const auto& feat : selected_features) {
        result.track_ids.push_back(feat->GetFeatureId());
    }
    
    // Compute rotation angle for summary
    Eigen::AngleAxisf angle_axis(R);
    float angle_deg = angle_axis.angle() * 180.0f / M_PI;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "    INITIALIZATION SUCCESS!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Frame pair: " << frame1->GetFrameId() 
              << " -> " << frame2->GetFrameId() 
              << " (separation: " << (frame2->GetFrameId() - frame1->GetFrameId()) << " frames)" << std::endl;
    std::cout << "  Triangulated points: " << num_triangulated << std::endl;
    std::cout << "  Rotation: " << std::fixed << std::setprecision(3) 
              << angle_deg << " degrees" << std::endl;
    std::cout << "  Translation direction: [" << std::fixed << std::setprecision(3)
              << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return true;
}

float Initializer::ComputeParallax(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2
) const {
    if (!frame1 || !frame2) {
        return 0.0f;
    }
    
    const auto& features1 = frame1->GetFeatures();
    const auto& features2 = frame2->GetFeatures();
    
    if (features1.empty() || features2.empty()) {
        return 0.0f;
    }
    
    // Find correspondences between two frames
    // Match by feature ID (tracked features have same ID)
    std::vector<float> parallaxes;
    parallaxes.reserve(features1.size());
    
    for (const auto& feat1 : features1) {
        // Find matching feature in frame2 by ID
        for (const auto& feat2 : features2) {
            if (feat1->GetFeatureId() == feat2->GetFeatureId()) {
                // Found correspondence
                cv::Point2f pt1 = feat1->GetPixelCoord();
                cv::Point2f pt2 = feat2->GetPixelCoord();
                
                // Compute Euclidean distance (parallax in pixels)
                float dx = pt2.x - pt1.x;
                float dy = pt2.y - pt1.y;
                float parallax = std::sqrt(dx * dx + dy * dy);
                
                parallaxes.push_back(parallax);
                break;
            }
        }
    }
    
    if (parallaxes.empty()) {
        return 0.0f;
    }
    
    // Compute median parallax (more robust than mean)
    std::sort(parallaxes.begin(), parallaxes.end());
    
    size_t mid = parallaxes.size() / 2;
    float median_parallax;
    
    if (parallaxes.size() % 2 == 0) {
        median_parallax = (parallaxes[mid - 1] + parallaxes[mid]) / 2.0f;
    } else {
        median_parallax = parallaxes[mid];
    }
    
    return median_parallax;
}

std::vector<std::shared_ptr<Feature>> Initializer::SelectFeaturesForInit(
    const std::vector<std::shared_ptr<Frame>>& frames
) const {
    std::vector<std::shared_ptr<Feature>> selected_features;
    
    if (frames.empty()) {
        std::cout << "[Initializer] SelectFeatures: No frames provided" << std::endl;
        return selected_features;
    }
    
    // Get features from the last frame (they have all the observation history)
    auto last_frame = frames.back();
    const auto& all_features = last_frame->GetFeatures();
    
    std::cout << "\n[Initializer] Feature Selection:" << std::endl;
    std::cout << "  Total features in last frame: " << all_features.size() << std::endl;
    
    // Filter by observation count
    std::vector<std::shared_ptr<Feature>> candidates;
    for (const auto& feat : all_features) {
        int obs_count = feat->GetObservationCount();
        if (obs_count >= m_min_observations) {
            candidates.push_back(feat);
        }
    }
    
    std::cout << "  Features with >= " << m_min_observations 
              << " observations: " << candidates.size() << std::endl;
    
    if (candidates.size() < static_cast<size_t>(m_min_features)) {
        std::cout << "  [WARNING] Not enough features! Need at least " 
                  << m_min_features << std::endl;
        return selected_features;
    }
    
    // Grid-based sampling for uniform distribution
    const int grid_cols = m_init_grid_cols;
    const int grid_rows = m_init_grid_rows;
    const int total_grids = grid_cols * grid_rows;
    
    // Get camera dimensions from config
    const int img_width = m_camera_width;
    const int img_height = m_camera_height;
    
    const float cell_width = static_cast<float>(img_width) / grid_cols;
    const float cell_height = static_cast<float>(img_height) / grid_rows;
    
    // Assign candidates to grid cells
    std::vector<std::vector<std::shared_ptr<Feature>>> grid(total_grids);
    
    for (const auto& feat : candidates) {
        cv::Point2f pt = feat->GetPixelCoord();
        int col = static_cast<int>(pt.x / cell_width);
        int row = static_cast<int>(pt.y / cell_height);
        
        // Clamp to valid range
        col = std::max(0, std::min(col, grid_cols - 1));
        row = std::max(0, std::min(row, grid_rows - 1));
        
        int grid_idx = row * grid_cols + col;
        grid[grid_idx].push_back(feat);
    }
    
    // Count non-empty cells
    int non_empty_cells = 0;
    for (const auto& cell : grid) {
        if (!cell.empty()) {
            non_empty_cells++;
        }
    }
    
    std::cout << "  Grid distribution: " << grid_cols << "x" << grid_rows 
              << " (" << non_empty_cells << "/" << total_grids << " cells occupied)" << std::endl;
    
    // Sample features from each cell
    const int max_per_cell = 5;  // Maximum features per cell
    
    for (int i = 0; i < total_grids; ++i) {
        if (grid[i].empty()) continue;
        
        // Sort by observation count (prefer longer tracks)
        std::sort(grid[i].begin(), grid[i].end(),
            [](const std::shared_ptr<Feature>& a, const std::shared_ptr<Feature>& b) {
                return a->GetObservationCount() > b->GetObservationCount();
            });
        
        // Take top features from this cell
        int count = std::min(max_per_cell, static_cast<int>(grid[i].size()));
        for (int j = 0; j < count; ++j) {
            selected_features.push_back(grid[i][j]);
        }
    }
    
    // Print observation statistics
    if (!selected_features.empty()) {
        int min_obs = selected_features[0]->GetObservationCount();
        int max_obs = selected_features[0]->GetObservationCount();
        int total_obs = 0;
        
        for (const auto& feat : selected_features) {
            int obs = feat->GetObservationCount();
            min_obs = std::min(min_obs, obs);
            max_obs = std::max(max_obs, obs);
            total_obs += obs;
        }
        
        float avg_obs = static_cast<float>(total_obs) / selected_features.size();
        
        std::cout << "  Selected features: " << selected_features.size() << std::endl;
        std::cout << "  Observation stats: min=" << min_obs 
                  << ", max=" << max_obs 
                  << ", avg=" << std::fixed << std::setprecision(1) << avg_obs << std::endl;
    }
    
    std::cout << std::endl;
    
    return selected_features;
}

bool Initializer::SelectFramePair(
    const std::vector<std::shared_ptr<Feature>>& features,
    std::shared_ptr<Frame>& frame1,
    std::shared_ptr<Frame>& frame2
) const {
    if (features.empty()) {
        return false;
    }
    
    // Use first and last observation frames from the first feature
    // (all features should have similar observation windows)
    const auto& observations = features[0]->GetObservations();
    
    if (observations.size() < 2) {
        return false;
    }
    
    frame1 = observations.front().frame;
    frame2 = observations.back().frame;
    
    std::cout << "[Initializer] Selected frame pair:" << std::endl;
    std::cout << "  Frame1 ID: " << frame1->GetFrameId() << std::endl;
    std::cout << "  Frame2 ID: " << frame2->GetFrameId() << std::endl;
    std::cout << "  Frame separation: " << (frame2->GetFrameId() - frame1->GetFrameId()) << " frames" << std::endl;
    
    return true;
}

bool Initializer::ComputeEssentialMatrix(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    Eigen::Matrix3f& E,
    std::vector<bool>& inlier_mask
) const {
    if (bearings1.size() != bearings2.size() || bearings1.size() < 5) {
        std::cout << "[Initializer] ComputeEssentialMatrix: Need at least 5 correspondences" << std::endl;
        return false;
    }
    
    const size_t num_points = bearings1.size();
    inlier_mask.resize(num_points, false);
    
    std::cout << "\n[Initializer] Computing Essential Matrix:" << std::endl;
    std::cout << "  Input correspondences: " << num_points << std::endl;
    std::cout << "  RANSAC iterations: " << m_ransac_iterations << std::endl;
    std::cout << "  RANSAC threshold: " << std::scientific << std::setprecision(3) 
              << m_ransac_threshold << " radians (" 
              << std::fixed << std::setprecision(2) << (m_ransac_threshold * 180.0f / M_PI) 
              << " degrees)" << std::endl;
    
    // Debug: Print some sample bearing vectors
    std::cout << "  Sample bearings:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), num_points); ++i) {
        std::cout << "    [" << i << "] b1=" << bearings1[i].transpose() 
                  << ", b2=" << bearings2[i].transpose() << std::endl;
    }
    
    // RANSAC parameters
    const int min_samples = 8;  // Use 8-point algorithm (more stable than 5-point)
    int best_inliers = 0;
    Eigen::Matrix3f best_E = Eigen::Matrix3f::Identity();
    std::vector<bool> best_mask(num_points, false);
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_points - 1);
    
    // RANSAC loop
    for (int iter = 0; iter < m_ransac_iterations; ++iter) {
        // 1. Randomly sample 5 points
        std::vector<int> sample_indices;
        sample_indices.reserve(min_samples);
        
        while (sample_indices.size() < static_cast<size_t>(min_samples)) {
            int idx = dis(gen);
            // Check for duplicate
            if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                sample_indices.push_back(idx);
            }
        }
        
        // 2. Compute Essential matrix from 8 points using 8-point algorithm
        std::vector<Eigen::Vector3f> sample_bearings1, sample_bearings2;
        for (int idx : sample_indices) {
            sample_bearings1.push_back(bearings1[idx]);
            sample_bearings2.push_back(bearings2[idx]);
        }
        
        // 8-point algorithm
        Eigen::MatrixXf A(min_samples, 9);  // Use 8 samples
        for (size_t i = 0; i < sample_bearings1.size(); ++i) {
            const auto& b1 = sample_bearings1[i];
            const auto& b2 = sample_bearings2[i];
            
            // Essential matrix constraint: b2^T * E * b1 = 0
            A(i, 0) = b2.x() * b1.x();
            A(i, 1) = b2.x() * b1.y();
            A(i, 2) = b2.x() * b1.z();
            A(i, 3) = b2.y() * b1.x();
            A(i, 4) = b2.y() * b1.y();
            A(i, 5) = b2.y() * b1.z();
            A(i, 6) = b2.z() * b1.x();
            A(i, 7) = b2.z() * b1.y();
            A(i, 8) = b2.z() * b1.z();
        }
        
        // Solve using SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXf e_vec = svd.matrixV().col(8);
        
        Eigen::Matrix3f E_candidate;
        E_candidate << e_vec(0), e_vec(1), e_vec(2),
                       e_vec(3), e_vec(4), e_vec(5),
                       e_vec(6), e_vec(7), e_vec(8);
        
        // Enforce Essential matrix constraint: two equal singular values, one zero
        Eigen::JacobiSVD<Eigen::Matrix3f> svd_E(E_candidate, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f singular_values = svd_E.singularValues();
        
        // Set singular values to [1, 1, 0]
        float sigma = (singular_values(0) + singular_values(1)) / 2.0f;
        Eigen::Vector3f new_singular_values(sigma, sigma, 0.0f);
        
        E_candidate = svd_E.matrixU() * new_singular_values.asDiagonal() * svd_E.matrixV().transpose();
        
        // 3. Count inliers
        std::vector<bool> current_mask(num_points, false);
        int num_inliers = 0;
        std::vector<float> errors;
        errors.reserve(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            const auto& b1 = bearings1[i];
            const auto& b2 = bearings2[i];
            
            // Compute Sampson error (first-order geometric error)
            float error = std::abs(b2.transpose() * E_candidate * b1);
            errors.push_back(error);
            
            if (error < m_ransac_threshold) {
                current_mask[i] = true;
                num_inliers++;
            }
        }
        
        // Debug: Print error statistics for first few iterations
        if (iter < 3) {
            std::sort(errors.begin(), errors.end());
            std::cout << "  Iter " << iter << " errors (rad): min=" << std::scientific << std::setprecision(3) << errors[0] 
                      << ", median=" << errors[errors.size()/2]
                      << ", max=" << errors.back()
                      << ", threshold=" << m_ransac_threshold
                      << ", inliers=" << num_inliers << "/" << num_points << std::endl;
        }
        
        // 4. Update best model
        if (num_inliers > best_inliers) {
            best_inliers = num_inliers;
            best_E = E_candidate;
            best_mask = current_mask;
        }
    }
    
    // Check if we have enough inliers
    float inlier_ratio = static_cast<float>(best_inliers) / num_points;
    
    std::cout << "  Best inliers: " << best_inliers << "/" << num_points 
              << " (" << std::fixed << std::setprecision(1) << (inlier_ratio * 100) << "%)" << std::endl;
    
    if (inlier_ratio < m_min_inlier_ratio) {
        std::cout << "  [FAIL] Inlier ratio too low! Minimum required: " 
                  << (m_min_inlier_ratio * 100) << "%" << std::endl;
        return false;
    }
    
    // Refine Essential matrix using all inliers
    Eigen::MatrixXf A_inliers(best_inliers, 9);
    int row = 0;
    for (size_t i = 0; i < num_points; ++i) {
        if (best_mask[i]) {
            const auto& b1 = bearings1[i];
            const auto& b2 = bearings2[i];
            
            A_inliers(row, 0) = b2.x() * b1.x();
            A_inliers(row, 1) = b2.x() * b1.y();
            A_inliers(row, 2) = b2.x() * b1.z();
            A_inliers(row, 3) = b2.y() * b1.x();
            A_inliers(row, 4) = b2.y() * b1.y();
            A_inliers(row, 5) = b2.y() * b1.z();
            A_inliers(row, 6) = b2.z() * b1.x();
            A_inliers(row, 7) = b2.z() * b1.y();
            A_inliers(row, 8) = b2.z() * b1.z();
            row++;
        }
    }
    
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_refine(A_inliers, Eigen::ComputeFullV);
    Eigen::VectorXf e_vec_refined = svd_refine.matrixV().col(8);
    
    E << e_vec_refined(0), e_vec_refined(1), e_vec_refined(2),
         e_vec_refined(3), e_vec_refined(4), e_vec_refined(5),
         e_vec_refined(6), e_vec_refined(7), e_vec_refined(8);
    
    // Enforce Essential matrix constraint again
    Eigen::JacobiSVD<Eigen::Matrix3f> svd_E_final(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f singular_values = svd_E_final.singularValues();
    float sigma = (singular_values(0) + singular_values(1)) / 2.0f;
    Eigen::Vector3f new_singular_values(sigma, sigma, 0.0f);
    E = svd_E_final.matrixU() * new_singular_values.asDiagonal() * svd_E_final.matrixV().transpose();
    
    inlier_mask = best_mask;
    
    std::cout << "  [SUCCESS] Essential matrix computed" << std::endl;
    std::cout << "  E = \n" << E << std::endl;
    std::cout << std::endl;
    
    return true;
}

bool Initializer::RecoverPose(
    const Eigen::Matrix3f& E,
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t,
    const std::vector<bool>& inlier_mask
) const {
    std::cout << "\n[Initializer] Recovering Pose from Essential Matrix" << std::endl;
    
    // 1. SVD decomposition: E = U * Σ * V^T
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    std::cout << "  SVD decomposition complete" << std::endl;
    std::cout << "  Singular values: " << svd.singularValues().transpose() << std::endl;
    
    // 2. Define W matrix for rotation extraction
    Eigen::Matrix3f W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;
    
    // 3. Generate 4 pose candidates
    Eigen::Matrix3f R1 = U * W * V.transpose();
    Eigen::Matrix3f R2 = U * W.transpose() * V.transpose();
    Eigen::Vector3f t1 = U.col(2);  // Last column of U
    
    // Ensure proper rotation (det(R) = +1, not -1)
    if (R1.determinant() < 0) {
        R1 = -R1;
        t1 = -t1;
    }
    if (R2.determinant() < 0) {
        R2 = -R2;
    }
    
    // Normalize translation to unit vector
    t1.normalize();
    
    std::vector<std::pair<Eigen::Matrix3f, Eigen::Vector3f>> candidates(4);
    candidates[0] = {R1,  t1};
    candidates[1] = {R1, -t1};
    candidates[2] = {R2,  t1};
    candidates[3] = {R2, -t1};
    
    std::cout << "  Generated 4 pose candidates" << std::endl;
    
    // 4. Test each candidate with cheirality check
    int best_count = 0;
    int best_idx = -1;
    
    for (int i = 0; i < 4; ++i) {
        int good_points = TestPoseCandidate(
            candidates[i].first,
            candidates[i].second,
            bearings1,
            bearings2,
            inlier_mask
        );
        
        std::cout << "  Candidate " << (i+1) << ": " << good_points 
                  << " points pass cheirality check" << std::endl;
        
        if (good_points > best_count) {
            best_count = good_points;
            best_idx = i;
        }
    }
    
    if (best_idx < 0 || best_count < 50) {
        std::cout << "  [FAIL] Not enough valid points (need >= 50, got " 
                  << best_count << ")" << std::endl;
        return false;
    }
    
    // 5. Select best pose
    R = candidates[best_idx].first;
    t = candidates[best_idx].second;
    
    // Count total inliers for ratio
    int total_inliers = 0;
    for (bool inlier : inlier_mask) {
        if (inlier) total_inliers++;
    }
    
    float valid_ratio = static_cast<float>(best_count) / total_inliers;
    
    std::cout << "  [SUCCESS] Selected candidate " << (best_idx + 1) << std::endl;
    std::cout << "  Valid points: " << best_count << "/" << total_inliers 
              << " (" << std::fixed << std::setprecision(1) << (valid_ratio * 100) << "%)" << std::endl;
    
    // Print R with higher precision
    std::cout << "  R = " << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "    ";
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << R(i, j);
        }
        std::cout << std::endl;
    }
    
    // Print t with higher precision
    std::cout << "  t = ";
    for (int i = 0; i < 3; ++i) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6) << t(i);
    }
    std::cout << " (unit vector, ||t|| = " << std::fixed << std::setprecision(6) 
              << t.norm() << ")" << std::endl;
    
    // Compute rotation angle
    Eigen::AngleAxisf angle_axis(R);
    float angle_deg = angle_axis.angle() * 180.0f / M_PI;
    std::cout << "  Rotation angle: " << std::fixed << std::setprecision(3) 
              << angle_deg << " degrees" << std::endl;
    std::cout << "  Rotation axis: " << angle_axis.axis().transpose() << std::endl;
    
    std::cout << std::endl;
    
    return true;
}

int Initializer::TriangulatePoints(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    std::vector<Eigen::Vector3f>& points3d
) const {
    std::cout << "\n[Initializer] Triangulating 3D points" << std::endl;
    std::cout << "  Input correspondences: " << bearings1.size() << std::endl;
    
    points3d.clear();
    points3d.reserve(bearings1.size());
    
    int success_count = 0;
    std::vector<float> depths;
    depths.reserve(bearings1.size());
    
    for (size_t i = 0; i < bearings1.size(); ++i) {
        Eigen::Vector3f point3d;
        
        // Triangulate using mid-point method
        if (TriangulateSinglePoint(bearings1[i], bearings2[i], R, t, point3d)) {
            // Verify depth positivity (cheirality check)
            float depth1 = bearings1[i].dot(point3d);
            Eigen::Vector3f point3d_in_frame2 = R * point3d + t;
            float depth2 = bearings2[i].dot(point3d_in_frame2);
            
            if (depth1 > 0 && depth2 > 0) {
                points3d.push_back(point3d);
                depths.push_back(depth1);
                success_count++;
            } else {
                points3d.push_back(Eigen::Vector3f::Zero());  // Mark as invalid
            }
        } else {
            points3d.push_back(Eigen::Vector3f::Zero());  // Triangulation failed
        }
    }
    
    // Compute statistics
    if (!depths.empty()) {
        std::sort(depths.begin(), depths.end());
        float min_depth = depths.front();
        float median_depth = depths[depths.size() / 2];
        float max_depth = depths.back();
        
        std::cout << "  Successfully triangulated: " << success_count 
                  << "/" << bearings1.size() << std::endl;
        std::cout << "  Depth statistics (frame1):" << std::endl;
        std::cout << "    Min: " << std::fixed << std::setprecision(3) << min_depth << std::endl;
        std::cout << "    Median: " << median_depth << std::endl;
        std::cout << "    Max: " << max_depth << std::endl;
    } else {
        std::cout << "  [WARNING] No points successfully triangulated!" << std::endl;
    }
    
    return success_count;
}

bool Initializer::TriangulateSinglePoint(
    const Eigen::Vector3f& bearing1,
    const Eigen::Vector3f& bearing2,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    Eigen::Vector3f& point3d
) const {
    // R: rotation from frame1 to frame2 (rot_ref_to_cur)
    // t: translation from frame1 to frame2 (trans_ref_to_cur)
    
    // Transform translation to go from frame2 to frame1
    const Eigen::Vector3f trans_12 = -R.transpose() * t;
    // Transform bearing2 to frame1 coordinates
    const Eigen::Vector3f bearing2_in_frame1 = R.transpose() * bearing2;
    
    // Build the linear system: A * lambda = b
    // Ray1: p1 = λ1 * bearing1
    // Ray2: p2 = λ2 * bearing2_in_frame1 + trans_12
    
    Eigen::Matrix2f A;
    A(0, 0) = bearing1.dot(bearing1);
    A(1, 0) = bearing1.dot(bearing2_in_frame1);
    A(0, 1) = -A(1, 0);
    A(1, 1) = -bearing2_in_frame1.dot(bearing2_in_frame1);
    
    Eigen::Vector2f b;
    b(0) = bearing1.dot(trans_12);
    b(1) = bearing2_in_frame1.dot(trans_12);
    
    // Solve for lambda
    const Eigen::Vector2f lambda = A.inverse() * b;
    
    // Check if depths are positive
    if (lambda(0) < 0 || lambda(1) < 0) {
        return false;
    }
    
    // Compute 3D points on both rays
    const Eigen::Vector3f pt_1 = lambda(0) * bearing1;
    const Eigen::Vector3f pt_2 = lambda(1) * bearing2_in_frame1 + trans_12;
    
    // Return the mid-point (in frame1 coordinates)
    point3d = (pt_1 + pt_2) / 2.0f;
    
    return true;
}

int Initializer::TestPoseCandidate(
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const std::vector<bool>& inlier_mask
) const {
    int good_points = 0;
    
    for (size_t i = 0; i < bearings1.size(); ++i) {
        if (!inlier_mask[i]) continue;
        
        // Triangulate 3D point
        Eigen::Vector3f point3d;
        if (!TriangulateSinglePoint(bearings1[i], bearings2[i], R, t, point3d)) {
            continue;
        }
        
        // Cheirality check 1: Point must be in front of camera 1
        float depth1 = bearings1[i].dot(point3d);
        if (depth1 <= 0) continue;
        
        // Cheirality check 2: Point must be in front of camera 2
        Eigen::Vector3f point3d_in_frame2 = R * point3d + t;
        float depth2 = bearings2[i].dot(point3d_in_frame2);
        if (depth2 <= 0) continue;
        
        good_points++;
    }
    
    return good_points;
}

float Initializer::ComputeReprojectionError(
    const Eigen::Vector3f& point3d,
    const Eigen::Vector3f& bearing_observed,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t
) const {
    // point3d: 3D point in frame1 (reference) coordinates
    // R: rotation from frame1 to frame2
    // t: translation from frame1 to frame2
    
    // Transform 3D point to frame2 coordinate (correct transformation)
    Eigen::Vector3f point_in_frame2 = R * point3d + t;
    
    // Normalize to bearing vector
    Eigen::Vector3f bearing_projected = point_in_frame2.normalized();
    
    // Convert observed bearing to pixel coordinates (equirectangular projection)
    float lon_obs = std::atan2(bearing_observed.x(), bearing_observed.z());
    float lat_obs = -std::asin(bearing_observed.y());
    float u_obs = m_camera_width * (0.5f + lon_obs / (2.0f * M_PI));
    float v_obs = m_camera_height * (0.5f - lat_obs / M_PI);
    
    // Convert projected bearing to pixel coordinates
    float lon_proj = std::atan2(bearing_projected.x(), bearing_projected.z());
    float lat_proj = -std::asin(bearing_projected.y());
    float u_proj = m_camera_width * (0.5f + lon_proj / (2.0f * M_PI));
    float v_proj = m_camera_height * (0.5f - lat_proj / M_PI);
    
    // Handle equirectangular wraparound for longitude (±180 degrees)
    float du = u_obs - u_proj;
    // If difference is more than half the image width, wrap around
    if (du > m_camera_width / 2.0f) {
        du -= m_camera_width;
    } else if (du < -m_camera_width / 2.0f) {
        du += m_camera_width;
    }
    
    float dv = v_obs - v_proj;
    
    // Compute pixel error (Euclidean distance)
    float pixel_error = std::sqrt(du * du + dv * dv);
    
    return pixel_error;
}

bool Initializer::ValidateInitialization(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const std::vector<Eigen::Vector3f>& points3d,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    const std::vector<bool>& inlier_mask
) const {
    std::cout << "\n[Initializer] Validating initialization" << std::endl;
    
    int valid_count = 0;
    int total_inliers = 0;
    std::vector<float> errors_pixel;
    
    for (size_t i = 0; i < points3d.size(); ++i) {
        if (!inlier_mask[i]) continue;
        total_inliers++;
        
        // Skip invalid points (zero vector)
        if (points3d[i].norm() < 1e-6) continue;
        
        // Compute reprojection error in frame2 (now returns pixel error directly)
        float error_pixel = ComputeReprojectionError(points3d[i], bearings2[i], R, t);
        
        errors_pixel.push_back(error_pixel);
        
        if (error_pixel < m_max_reprojection_error) {
            valid_count++;
        }
    }
    
    if (errors_pixel.empty()) {
        std::cout << "  [FAIL] No valid points to validate!" << std::endl;
        return false;
    }
    
    // Compute statistics
    std::sort(errors_pixel.begin(), errors_pixel.end());
    
    float median_error_pixel = errors_pixel[errors_pixel.size() / 2];
    float max_error_pixel = errors_pixel.back();
    float mean_error_pixel = 0.0f;
    for (float e : errors_pixel) mean_error_pixel += e;
    mean_error_pixel /= errors_pixel.size();
    
    float valid_ratio = static_cast<float>(valid_count) / total_inliers;
    
    std::cout << "  Reprojection error statistics (pixels):" << std::endl;
    std::cout << "    Min: " << std::fixed << std::setprecision(2) 
              << errors_pixel.front() << " px" << std::endl;
    std::cout << "    Median: " << std::fixed << std::setprecision(2) 
              << median_error_pixel << " px" << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(2) 
              << mean_error_pixel << " px" << std::endl;
    std::cout << "    Max: " << std::fixed << std::setprecision(2) 
              << max_error_pixel << " px" << std::endl;
    std::cout << "    Threshold: " << m_max_reprojection_error << " px" << std::endl;
    std::cout << "  Valid points: " << valid_count << "/" << total_inliers 
              << " (" << std::fixed << std::setprecision(1) << (valid_ratio * 100) << "%)" << std::endl;
    
    // Validation check 1: Inlier ratio
    if (valid_ratio < m_min_inlier_ratio) {
        std::cout << "  [FAIL] Valid ratio too low! Required: >= " 
                  << std::fixed << std::setprecision(1) << (m_min_inlier_ratio * 100) << "%" << std::endl;
        return false;
    }
    
    // Validation check 2: Minimum number of points
    if (valid_count < m_min_features) {
        std::cout << "  [FAIL] Not enough valid points! Required: >= " 
                  << m_min_features << std::endl;
        return false;
    }
    
    std::cout << "  [SUCCESS] Validation passed!" << std::endl;
    return true;
}

} // namespace vio_360
