/**
 * @file      Initializer.cpp
 * @brief     Implementation of Initializer
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Initializer.h"
#include "ConfigUtils.h"
#include "Logger.h"
#include "LieUtils.h"
#include "optimization/Optimizer.h"
#include <algorithm>
#include <cmath>
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
}

void Initializer::Reset() {
    m_is_initialized = false;
}

bool Initializer::TryMonocularInitialization(
    const std::vector<std::shared_ptr<Frame>>& frames,
    InitializationResult& result
) {
    // Reset result
    result = InitializationResult();
    
    if (frames.size() < 2) {
        LOG_INFO("  [MonoInit] frames.size()={} < 2", frames.size());
        return false;
    }
    
    // 1. Select features with sufficient observations
    auto selected_features = SelectFeaturesForInit(frames);
    
    LOG_INFO("  [MonoInit] Selected {} features (min={})", selected_features.size(), m_min_features);
    
    if (selected_features.size() < static_cast<size_t>(m_min_features)) {
        LOG_INFO("  [MonoInit] Not enough features");
        return false;
    }
    
    // 2. Use window's first and last frames directly
    std::shared_ptr<Frame> frame1 = frames.front();
    std::shared_ptr<Frame> frame2 = frames.back();
    
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
    
    LOG_INFO("  [MonoInit] Valid bearing pairs: {}", bearings1.size());
    
    if (bearings1.size() < 5) {
        LOG_INFO("  [MonoInit] Too few bearings (<5)");
        return false;
    }
    
    // 4. Compute Essential matrix
    Eigen::Matrix3f E;
    std::vector<bool> inlier_mask;
    
    if (!ComputeEssentialMatrix(bearings1, bearings2, E, inlier_mask)) {
        LOG_WARN("  [MonoInit] Essential matrix computation failed");
        return false;
    }
    
    int E_inliers = std::count(inlier_mask.begin(), inlier_mask.end(), true);
    LOG_INFO("  [MonoInit] Essential matrix: {} inliers / {} total", E_inliers, inlier_mask.size());
    
    // 5. Recover pose (R, t) from Essential matrix
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    
    if (!RecoverPose(E, bearings1, bearings2, R, t, inlier_mask)) {
        LOG_WARN("  [MonoInit] Pose recovery failed");
        return false;
    }
    
    LOG_INFO("  [MonoInit] Pose recovered: t=({:.4f},{:.4f},{:.4f})", t.x(), t.y(), t.z());
    
    // 6. Triangulate all inlier points
    std::vector<Eigen::Vector3f> points3d;
    int num_triangulated = TriangulatePoints(bearings1, bearings2, R, t, points3d);
    
    LOG_INFO("  [MonoInit] Triangulated {} points (min={})", num_triangulated, m_min_features);
    
    if (num_triangulated < m_min_features) {
        LOG_WARN("  [MonoInit] Not enough triangulated points");
        return false;
    }
    
    // 7. Validate initialization quality
    float mean_reproj_error = 0.0f;
    if (!ValidateInitialization(bearings1, bearings2, points3d, R, t, inlier_mask, mean_reproj_error)) {
        LOG_WARN("  [MonoInit] Validation failed, mean_reproj={:.2f}px", mean_reproj_error);
        return false;
    }
    
    LOG_INFO("  [MonoInit] Validation passed, mean_reproj={:.2f}px", mean_reproj_error);
    
    // 8. Normalize scale so median depth = 10.0 (reasonable indoor/outdoor scale)
    float scale_factor = NormalizeScale(points3d, t);
    
    // 9. Set frame poses
    // Essential matrix gives us camera-to-camera transformation T_c1c2
    // T_c1c2 = [R | t]: transforms points from camera1 frame to camera2 frame
    // We need to convert camera poses to body poses using T_BC (camera-to-body)
    
    // Get extrinsic transformation
    Eigen::Matrix4f T_BC = frame1->GetTBC();  // camera-to-body (c → b)
    Eigen::Matrix4f T_CB = T_BC.inverse();    // body-to-camera (b → c)
    
    // Debug: Print T_BC to verify it's loaded correctly
    LOG_INFO("T_BC (camera→body):");
    LOG_INFO("  [{:.4f}, {:.4f}, {:.4f}, {:.4f}]", T_BC(0,0), T_BC(0,1), T_BC(0,2), T_BC(0,3));
    LOG_INFO("  [{:.4f}, {:.4f}, {:.4f}, {:.4f}]", T_BC(1,0), T_BC(1,1), T_BC(1,2), T_BC(1,3));
    LOG_INFO("  [{:.4f}, {:.4f}, {:.4f}, {:.4f}]", T_BC(2,0), T_BC(2,1), T_BC(2,2), T_BC(2,3));
    LOG_INFO("  [{:.4f}, {:.4f}, {:.4f}, {:.4f}]", T_BC(3,0), T_BC(3,1), T_BC(3,2), T_BC(3,3));
    
    // World frame = Body1 frame (body1 is at world origin)
    // T_wb1 = Identity
    Eigen::Matrix4f T_wb1 = Eigen::Matrix4f::Identity();
    frame1->SetTwb(T_wb1);
    frame1->SetKeyframe(true);
    
    // T_wc1 = T_wb1 * T_BC = T_BC (camera1 pose in world)
    Eigen::Matrix4f T_wc1 = T_wb1 * T_BC;
    
    // T_c1c2 = [R | t]: transforms points from camera1 to camera2 frame
    // T_c2c1 = T_c1c2^(-1): transforms points from camera2 to camera1 frame
    // Camera2 pose in world: T_wc2 = T_wc1 * T_c2c1
    Eigen::Matrix4f T_c1c2 = Eigen::Matrix4f::Identity();
    T_c1c2.block<3, 3>(0, 0) = R;
    T_c1c2.block<3, 1>(0, 3) = t;
    Eigen::Matrix4f T_c2c1 = T_c1c2.inverse();
    Eigen::Matrix4f T_wc2 = T_wc1 * T_c2c1;
    
    // Convert camera2 pose to body2 pose: T_wb2 = T_wc2 * T_CB
    Eigen::Matrix4f T_wb2 = T_wc2 * T_CB;
    frame2->SetTwb(T_wb2);
    frame2->SetKeyframe(true);
    
    // Debug: verify camera centers
    {
        Eigen::Matrix4f T_wc1_check = frame1->GetTwc();
        Eigen::Matrix4f T_wc2_check = frame2->GetTwc();
        Eigen::Vector3f C1 = T_wc1_check.block<3, 1>(0, 3);
        Eigen::Vector3f C2 = T_wc2_check.block<3, 1>(0, 3);
        LOG_INFO("Init poses: C1=({:.4f},{:.4f},{:.4f}), C2=({:.4f},{:.4f},{:.4f}), baseline={:.4f}",
                 C1.x(), C1.y(), C1.z(), C2.x(), C2.y(), C2.z(), (C1-C2).norm());
    }
    
    // 10. Transform points from camera1 frame to world (=body1) frame
    // points3d are in camera1 coordinates, need to transform to world
    // p_world = T_wc1 * p_c1 = T_BC * p_c1 (since T_wb1 = I)
    Eigen::Matrix3f R_wc1 = T_wc1.block<3, 3>(0, 0);
    Eigen::Vector3f t_wc1 = T_wc1.block<3, 1>(0, 3);
    for (auto& pt : points3d) {
        pt = R_wc1 * pt + t_wc1;
    }
    
    // 11. Create MapPoints and register to frames
    CreateMapPoints(frame1, frame2, points3d, selected_features, result);
    
    // 12. Success! Mark as initialized
    m_is_initialized = true;
    
    // 13. Populate result structure
    result.success = true;
    result.R = R;
    result.t = t;
    result.points3d = points3d;
    result.frame1_id = frame1->GetFrameId();
    result.frame2_id = frame2->GetFrameId();
    result.scale_factor = scale_factor;
    result.initialized_keyframes.push_back(frame1);
    result.initialized_keyframes.push_back(frame2);
    
    // Collect track IDs for the triangulated points
    result.track_ids.clear();
    result.track_ids.reserve(selected_features.size());
    for (const auto& feat : selected_features) {
        result.track_ids.push_back(feat->GetFeatureId());
    }
    
    // 12. Run Bundle Adjustment on two keyframes only
    LOG_INFO("Running BA on 2 keyframes with {} MapPoints...", result.initialized_mappoints.size());
    
    Optimizer optimizer;
    std::vector<std::shared_ptr<Frame>> keyframes = {frame1, frame2};
    
    // Log reprojection error BEFORE BA
    double reproj1_before = ComputeFrameReprojectionError(frame1);
    double reproj2_before = ComputeFrameReprojectionError(frame2);
    LOG_INFO("  Before BA - Frame {}: {:.2f} px, Frame {}: {:.2f} px, avg: {:.2f} px",
             frame1->GetFrameId(), reproj1_before, 
             frame2->GetFrameId(), reproj2_before,
             (reproj1_before + reproj2_before) / 2.0);
    
    BAResult ba_result = optimizer.RunFullBA(keyframes);
    
    if (ba_result.success) {
        LOG_INFO("BA success: {} inliers, {} outliers, cost {:.4f} -> {:.4f}",
                 ba_result.num_inliers, ba_result.num_outliers,
                 ba_result.initial_cost, ba_result.final_cost);
        
        // Log reprojection error AFTER BA
        double reproj1_after = ComputeFrameReprojectionError(frame1);
        double reproj2_after = ComputeFrameReprojectionError(frame2);
        LOG_INFO("  After BA  - Frame {}: {:.2f} px, Frame {}: {:.2f} px, avg: {:.2f} px",
                 frame1->GetFrameId(), reproj1_after,
                 frame2->GetFrameId(), reproj2_after,
                 (reproj1_after + reproj2_after) / 2.0);
    } else {
        LOG_WARN("BA failed or not converged");
    }
    
    // Compute rotation angle for summary
    Eigen::AngleAxisf angle_axis(R);
    float angle_deg = angle_axis.angle() * 180.0f / M_PI;
    
    LOG_INFO("Init success: frames {}->{}, {} pts, scale={:.4f}, rot={:.1f}°, mean_reproj={:.2f}px",
             frame1->GetFrameId(), frame2->GetFrameId(),
             result.initialized_mappoints.size(), scale_factor, angle_deg, mean_reproj_error);
    
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
        return selected_features;
    }
    
    // Get features from the last frame (they have all the observation history)
    auto last_frame = frames.back();
    const auto& all_features = last_frame->GetFeatures();
    
    // Filter by observation count
    std::vector<std::shared_ptr<Feature>> candidates;
    for (const auto& feat : all_features) {
        int obs_count = feat->GetObservationCount();
        if (obs_count >= m_min_observations) {
            candidates.push_back(feat);
        }
    }
    
    if (candidates.size() < static_cast<size_t>(m_min_features)) {
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
    
    return true;
}

bool Initializer::ComputeEssentialMatrix(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    Eigen::Matrix3f& E,
    std::vector<bool>& inlier_mask
) const {
    if (bearings1.size() != bearings2.size() || bearings1.size() < 5) {
        return false;
    }
    
    const size_t num_points = bearings1.size();
    inlier_mask.resize(num_points, false);
    
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
        
        // Debug: print error distribution on first iteration
        if (iter == 0) {
            std::sort(errors.begin(), errors.end());
            LOG_INFO("  [Essential] Iter0 errors: min={:.4f}, median={:.4f}, max={:.4f}, thresh={:.4f}",
                     errors.front(), errors[errors.size()/2], errors.back(), m_ransac_threshold);
        }
        
        // 4. Update best model
        if (num_inliers > best_inliers) {
            best_inliers = num_inliers;
            best_E = E_candidate;
            best_mask = current_mask;
        }
    }
    
    // Check if we have enough inliers (use absolute count, not ratio)
    LOG_INFO("  [Essential] best_inliers={}, num_points={}, min_required={}", 
             best_inliers, num_points, m_min_features);
    
    if (best_inliers < m_min_features) {
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
    // 1. SVD decomposition: E = U * Σ * V^T
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
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
        
        LOG_INFO("  [RecoverPose] Candidate {}: good_points={}", i, good_points);
        
        if (good_points > best_count) {
            best_count = good_points;
            best_idx = i;
        }
    }
    
    LOG_INFO("  [RecoverPose] Best: idx={}, count={}, min_required={}", best_idx, best_count, m_min_features);
    
    if (best_idx < 0 || best_count < m_min_features) {
        return false;
    }
    
    // 5. Select best pose
    R = candidates[best_idx].first;
    t = candidates[best_idx].second;
    
    return true;
}

int Initializer::TriangulatePoints(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    std::vector<Eigen::Vector3f>& points3d
) const {
    points3d.clear();
    points3d.reserve(bearings1.size());
    
    int success_count = 0;
    
    for (size_t i = 0; i < bearings1.size(); ++i) {
        Eigen::Vector3f point3d;
        
        // Triangulate using mid-point method
        if (TriangulateSinglePoint(bearings1[i], bearings2[i], R, t, point3d)) {
            // For 360 cameras, skip cheirality check (depth can be negative)
            // The validity is already checked in TriangulateSinglePoint
            points3d.push_back(point3d);
            success_count++;
        } else {
            points3d.push_back(Eigen::Vector3f::Zero());  // Triangulation failed
        }
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
    
    // Check for degenerate matrix
    float det = A(0,0) * A(1,1) - A(0,1) * A(1,0);
    if (std::abs(det) < 1e-10f) {
        return false;
    }
    
    // Solve for lambda
    const Eigen::Vector2f lambda = A.inverse() * b;
    
    // For equirectangular cameras, skip depth positive check
    // (depth_is_positive = false for bearing_vector initializer)
    // Only check for finite values
    if (!std::isfinite(lambda(0)) || !std::isfinite(lambda(1))) {
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
    int tri_fail = 0;
    int reproj_fail = 0;
    
    // For equirectangular, use reprojection error to validate
    // instead of cheirality check
    
    for (size_t i = 0; i < bearings1.size(); ++i) {
        if (!inlier_mask[i]) continue;
        
        // Triangulate 3D point (in frame1 coordinates)
        Eigen::Vector3f point3d;
        if (!TriangulateSinglePoint(bearings1[i], bearings2[i], R, t, point3d)) {
            tri_fail++;
            continue;
        }
        
        // Check reprojection error in frame1 (reference)
        float err_ref = ComputeReprojectionErrorInFrame(point3d, bearings1[i]);
        
        // Check reprojection error in frame2 (current)
        Eigen::Vector3f point_in_frame2 = R * point3d + t;
        float err_cur = ComputeReprojectionErrorInFrame(point_in_frame2, bearings2[i]);
        
        // Use reprojection error threshold
        constexpr float kReprojErrThr = 5.0f;  // pixels
        if (err_ref < kReprojErrThr && err_cur < kReprojErrThr) {
            good_points++;
        } else {
            reproj_fail++;
        }
    }
    
    // Debug first candidate only
    static bool first_debug = true;
    if (first_debug) {
        int total_inliers = std::count(inlier_mask.begin(), inlier_mask.end(), true);
        LOG_INFO("    [TestPose Debug] inliers={}, tri_fail={}, reproj_fail={}, good={}", 
                 total_inliers, tri_fail, reproj_fail, good_points);
        first_debug = false;
    }
    
    return good_points;
}

float Initializer::ComputeReprojectionErrorInFrame(
    const Eigen::Vector3f& point3d_in_frame,
    const Eigen::Vector3f& bearing_observed
) const {
    // Project 3D point to pixel coordinates
    // point3d_in_frame: 3D point already in the frame's coordinate system
    // bearing_observed: observed bearing in the same frame (normalized)
    
    float L = point3d_in_frame.norm();
    if (L < 1e-6f) {
        return 1000.0f;  // Invalid point
    }
    
    // cam_project: theta = atan2(x, z), phi = -asin(y/L)
    // u = cols * (0.5 + theta/(2*pi)), v = rows * (0.5 - phi/pi)
    
    // Observed bearing -> pixel (bearing is already normalized)
    float theta_obs = std::atan2(bearing_observed.x(), bearing_observed.z());
    float phi_obs = -std::asin(std::clamp(bearing_observed.y(), -1.0f, 1.0f));
    float u_obs = m_camera_width * (0.5f + theta_obs / (2.0f * M_PI));
    float v_obs = m_camera_height * (0.5f - phi_obs / M_PI);
    
    // Projected point -> pixel (normalize then project)
    Eigen::Vector3f bearing_proj = point3d_in_frame / L;
    float theta_proj = std::atan2(bearing_proj.x(), bearing_proj.z());
    float phi_proj = -std::asin(std::clamp(bearing_proj.y(), -1.0f, 1.0f));
    float u_proj = m_camera_width * (0.5f + theta_proj / (2.0f * M_PI));
    float v_proj = m_camera_height * (0.5f - phi_proj / M_PI);
    
    // Simple pixel error (no wrap-around)
    float du = u_obs - u_proj;
    float dv = v_obs - v_proj;
    
    return std::sqrt(du * du + dv * dv);
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
    
    // Transform 3D point to frame2 coordinate
    Eigen::Vector3f point_in_frame2 = R * point3d + t;
    
    return ComputeReprojectionErrorInFrame(point_in_frame2, bearing_observed);
}

bool Initializer::ValidateInitialization(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const std::vector<Eigen::Vector3f>& points3d,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    const std::vector<bool>& inlier_mask,
    float& mean_error
) const {
    // Validation:
    // Check reprojection error in BOTH frames (ref and cur)
    
    int valid_count = 0;
    int total_inliers = 0;
    int skip_zero = 0;
    int skip_ref = 0;
    int skip_cur = 0;
    std::vector<float> errors_pixel;
    float error_sum = 0.0f;
    
    const float reproj_err_thr_sq = m_max_reprojection_error * m_max_reprojection_error;
    
    for (size_t i = 0; i < points3d.size(); ++i) {
        if (!inlier_mask[i]) continue;
        total_inliers++;
        
        // Skip invalid points (zero vector - triangulation failed)
        if (points3d[i].norm() < 1e-6) {
            skip_zero++;
            continue;
        }
        
        const Eigen::Vector3f& pt = points3d[i];
        
        // Check reprojection error in BOTH frames
        
        // 1. Reprojection error in frame1 (reference)
        float error_ref = ComputeReprojectionErrorInFrame(pt, bearings1[i]);
        
        // DEBUG: Show first few points
        if (total_inliers <= 5) {
            Eigen::Vector3f b1 = bearings1[i];
            Eigen::Vector3f pt_dir = pt.normalized();
            float dot = pt_dir.dot(b1);
            LOG_INFO("  [DBG] pt[{}]: pt=({:.3f},{:.3f},{:.3f}), b1=({:.3f},{:.3f},{:.3f}), dot={:.3f}, err_ref={:.2f}px",
                     i, pt.x(), pt.y(), pt.z(), b1.x(), b1.y(), b1.z(), dot, error_ref);
        }
        
        if (error_ref > m_max_reprojection_error) {
            skip_ref++;
            continue;
        }
        
        // 2. Reprojection error in frame2 (current)
        // Transform point to frame2: P_cur = R * P_ref + t
        Eigen::Vector3f point_in_frame2 = R * pt + t;
        float error_cur = ComputeReprojectionErrorInFrame(point_in_frame2, bearings2[i]);
        
        if (error_cur > m_max_reprojection_error) {
            skip_cur++;
            continue;
        }
        
        // Use max of both errors for statistics
        float max_error = std::max(error_ref, error_cur);
        errors_pixel.push_back(max_error);
        error_sum += max_error;
        valid_count++;
    }
    
    LOG_INFO("  [Validate] Stats: total_inliers={}, skip_zero={}, skip_ref={}, skip_cur={}, valid={}",
             total_inliers, skip_zero, skip_ref, skip_cur, valid_count);
    
    if (errors_pixel.empty()) {
        LOG_WARN("  [Validate] FAIL: No valid points! total_inliers={}", total_inliers);
        mean_error = 0.0f;
        return false;
    }
    
    // Compute mean error
    mean_error = error_sum / errors_pixel.size();
    
    // Compute statistics
    std::sort(errors_pixel.begin(), errors_pixel.end());
    float valid_ratio = static_cast<float>(valid_count) / total_inliers;
    
    LOG_INFO("  [Validate] valid={}/{} ({:.1f}%), mean_err={:.2f}px, median={:.2f}px",
             valid_count, total_inliers, valid_ratio * 100.0f, mean_error,
             errors_pixel[errors_pixel.size() / 2]);
    LOG_INFO("  [Validate] thresholds: max_reproj={:.2f}px, min_ratio={:.2f}, min_features={}",
             m_max_reprojection_error, m_min_inlier_ratio, m_min_features);
    
    // DEBUG: Show error distribution
    size_t n = errors_pixel.size();
    LOG_INFO("  [Validate] Error distribution: min={:.2f}, 25%={:.2f}, 50%={:.2f}, 75%={:.2f}, 90%={:.2f}, 95%={:.2f}, max={:.2f}",
             errors_pixel[0], errors_pixel[n/4], errors_pixel[n/2], 
             errors_pixel[3*n/4], errors_pixel[std::min(n-1, (size_t)(n*0.9))],
             errors_pixel[std::min(n-1, (size_t)(n*0.95))], errors_pixel[n-1]);
    
    // Validation check: Minimum number of points (ratio check removed for 360 cameras)
    if (valid_count < m_min_features) {
        LOG_WARN("  [Validate] FAIL: valid_count {} < min_features {}", valid_count, m_min_features);
        return false;
    }
    
    return true;
}

float Initializer::NormalizeScale(
    std::vector<Eigen::Vector3f>& points3d,
    Eigen::Vector3f& t
) const {
    if (points3d.empty()) {
        return 1.0f;
    }
    
    // Collect depths (Z-coordinate in frame1 = forward direction)
    // For 360 camera, depth is the distance from camera center
    std::vector<float> depths;
    depths.reserve(points3d.size());
    
    for (const auto& pt : points3d) {
        // Skip invalid points (zero vector)
        if (pt.norm() < 1e-6f) continue;
        
        // For 360 camera: depth = distance from origin
        float depth = pt.norm();
        if (depth > 0.01f) {  // Valid depth
            depths.push_back(depth);
        }
    }
    
    if (depths.empty()) {
        return 1.0f;
    }
    
    // Compute median depth
    std::sort(depths.begin(), depths.end());
    float median_depth;
    size_t mid = depths.size() / 2;
    if (depths.size() % 2 == 0) {
        median_depth = (depths[mid - 1] + depths[mid]) / 2.0f;
    } else {
        median_depth = depths[mid];
    }
    
    // Scale factor to normalize median depth to 1.0 (unit scale for IMU initialization)
    const float target_median_depth = 1.0f;
    float scale_factor = target_median_depth / median_depth;
    
    // Scale all 3D points
    for (auto& pt : points3d) {
        pt *= scale_factor;
    }
    
    // Scale translation vector
    t *= scale_factor;
    
    return scale_factor;
}

void Initializer::CreateMapPoints(
    std::shared_ptr<Frame> frame1,
    std::shared_ptr<Frame> frame2,
    const std::vector<Eigen::Vector3f>& points3d,
    const std::vector<std::shared_ptr<Feature>>& selected_features,
    InitializationResult& result
) {
    // Initialize map_points vectors in frames
    frame1->InitializeMapPoints();
    frame2->InitializeMapPoints();
    
    const auto& features1 = frame1->GetFeatures();
    const auto& features2 = frame2->GetFeatures();
    
    result.initialized_mappoints.clear();
    
    int created_count = 0;
    
    for (size_t i = 0; i < selected_features.size() && i < points3d.size(); ++i) {
        const auto& feat = selected_features[i];
        const Eigen::Vector3f& point_world = points3d[i];
        
        // Skip invalid points
        if (point_world.norm() < 1e-6f) continue;
        
        int feature_id = feat->GetFeatureId();
        
        // Find feature indices in both frames
        int feat_idx1 = -1, feat_idx2 = -1;
        
        for (size_t j = 0; j < features1.size(); ++j) {
            if (features1[j]->GetFeatureId() == feature_id) {
                feat_idx1 = static_cast<int>(j);
                break;
            }
        }
        
        for (size_t j = 0; j < features2.size(); ++j) {
            if (features2[j]->GetFeatureId() == feature_id) {
                feat_idx2 = static_cast<int>(j);
                break;
            }
        }
        
        // Skip if not found in both frames
        if (feat_idx1 < 0 || feat_idx2 < 0) continue;
        
        // Create MapPoint
        auto mp = std::make_shared<MapPoint>(point_world);
        
        // Register to frames
        frame1->SetMapPoint(feat_idx1, mp);
        frame2->SetMapPoint(feat_idx2, mp);
        
        // Add observations
        mp->AddObservation(frame1, feat_idx1);
        mp->AddObservation(frame2, feat_idx2);
        
        result.initialized_mappoints.push_back(mp);
        created_count++;
    }
}

void Initializer::InterpolateIntermediateFrames(
    const std::vector<std::shared_ptr<Frame>>& frames,
    const std::vector<std::shared_ptr<MapPoint>>& mappoints
) {
    if (frames.size() < 3) {
        return;  // No intermediate frames to interpolate
    }
    
    // Get keyframe poses (first and last)
    auto frame1 = frames.front();
    auto frame2 = frames.back();
    Eigen::Matrix4f T1 = frame1->GetTwb();
    Eigen::Matrix4f T2 = frame2->GetTwb();
    
    int num_frames = static_cast<int>(frames.size());
    int num_intermediate = num_frames - 2;
    
    // Interpolate intermediate frames and set as keyframes
    for (int i = 1; i < num_frames - 1; ++i) {
        auto& frame = frames[i];
        
        // Interpolation parameter: t in [0, 1]
        float t = static_cast<float>(i) / static_cast<float>(num_frames - 1);
        
        // Interpolate pose using SLERP for rotation, LERP for translation
        Eigen::Matrix4f T_interp = InterpolatePose(T1, T2, t);
        frame->SetTwb(T_interp);
        frame->SetKeyframe(true);  // Mark as keyframe
    }
    
    // Register MapPoint observations in intermediate frames
    int total_obs = 0;
    for (const auto& mp : mappoints) {
        Eigen::Vector3f point_world = mp->GetPosition();
        
        // For each intermediate frame
        for (int i = 1; i < num_frames - 1; ++i) {
            auto& frame = frames[i];
            const auto& features = frame->GetFeatures();
            
            // Find if this MapPoint's feature is tracked in this frame
            // MapPoint has observations from frame1 and frame2
            // We need to find the corresponding feature in intermediate frames by track_id
            
            const auto& observations = mp->GetObservations();
            if (observations.empty()) continue;
            
            // Get the feature_id (track_id) from one of the observations
            int track_id = -1;
            for (const auto& obs : observations) {
                auto obs_frame = obs.frame.lock();
                if (obs_frame && obs.feature_index >= 0 && 
                    obs.feature_index < static_cast<int>(obs_frame->GetFeatures().size())) {
                    track_id = obs_frame->GetFeatures()[obs.feature_index]->GetFeatureId();
                    break;
                }
            }
            
            if (track_id < 0) continue;
            
            // Find feature with same track_id in current frame
            for (size_t j = 0; j < features.size(); ++j) {
                if (features[j]->GetFeatureId() == track_id) {
                    // Check if this point is visible from this frame (positive depth)
                    Eigen::Matrix4f Twb = frame->GetTwb();
                    Eigen::Matrix4f Tbw = Twb.inverse();
                    Eigen::Vector3f point_cam = Tbw.block<3, 3>(0, 0) * point_world + 
                                                 Tbw.block<3, 1>(0, 3);
                    
                    // For 360 camera, depth is the distance from camera center
                    float depth = point_cam.norm();
                    
                    if (depth > 0.01f) {  // Valid depth
                        // Register observation
                        frame->SetMapPoint(static_cast<int>(j), mp);
                        mp->AddObservation(frame, static_cast<int>(j));
                        total_obs++;
                    }
                    break;
                }
            }
        }
    }
    
    LOG_INFO("Interpolated {} intermediate frames, added {} MapPoint observations",
             num_intermediate, total_obs);
}

double Initializer::ComputeFrameReprojectionError(const std::shared_ptr<Frame>& frame) const {
    if (!frame) return 0.0;
    
    // Get camera extrinsics
    Eigen::Matrix4f Twb = frame->GetTwb();
    Eigen::Matrix4f Tcb = frame->GetTCB();
    
    // Compute Tcw = Tcb * Tbw = Tcb * Twb^(-1)
    Eigen::Matrix4f Tbw = Twb.inverse();
    Eigen::Matrix4f Tcw = Tcb * Tbw;
    Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
    Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
    
    double cols = static_cast<double>(frame->GetWidth());
    double rows = static_cast<double>(frame->GetHeight());
    
    const auto& features = frame->GetFeatures();
    double total_error = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < features.size(); ++i) {
        auto mp = frame->GetMapPoint(static_cast<int>(i));
        if (!mp || mp->IsBad()) continue;
        
        const auto& feature = features[i];
        if (!feature || !feature->IsValid()) continue;
        
        // Get observed pixel coordinate
        cv::Point2f obs_pt = feature->GetPixelCoord();
        double obs_u = obs_pt.x;
        double obs_v = obs_pt.y;
        
        // Get MapPoint position in world frame
        Eigen::Vector3f pw = mp->GetPosition();
        
        // Transform to camera frame
        Eigen::Vector3f pc = Rcw * pw + tcw;
        double pcx = pc.x();
        double pcy = pc.y();
        double pcz = pc.z();
        double L = pc.norm();
        
        if (L < 1e-6) continue;
        
        // Equirectangular projection
        // Camera frame: X-right, Y-down, Z-forward
        // theta = atan2(x, z), phi = -asin(y/L)
        double theta = std::atan2(pcx, pcz);  // [-π, π]
        double phi = -std::asin(pcy / L);     // [-π/2, π/2]
        
        // u = cols * (0.5 + theta/(2π)), v = rows * (0.5 - phi/π)
        double proj_u = cols * (0.5 + theta / (2.0 * M_PI));
        double proj_v = rows * (0.5 - phi / M_PI);
        
        // Compute pixel error
        double du = obs_u - proj_u;
        double dv = obs_v - proj_v;
        
        // Handle wraparound for u (horizontal)
        if (du > cols / 2.0) du -= cols;
        if (du < -cols / 2.0) du += cols;
        
        double error = std::sqrt(du * du + dv * dv);
        total_error += error;
        count++;
    }
    
    return (count > 0) ? (total_error / count) : 0.0;
}

} // namespace vio_360
