/**
 * @file      Optimizer.cpp
 * @brief     Implementation of PnP and Bundle Adjustment optimizers
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Optimizer.h"
#include "Factors.h"
#include "database/Frame.h"
#include "database/MapPoint.h"
#include "database/Feature.h"
#include "database/Camera.h"
#include "util/Logger.h"
#include "util/LieUtils.h"

#include <set>

namespace vio_360 {

Optimizer::Optimizer()
    : m_huber_delta(1.0)
    , m_pixel_noise_std(1.0)
    , m_max_iterations(50)
    , m_chi2_threshold(9.21)    // Chi-square 99% for 2 DOF (more permissive)
    , m_camera(nullptr)
    , m_boundary_margin(20) {
}

void Optimizer::SetCamera(std::shared_ptr<Camera> camera, int boundary_margin) {
    m_camera = camera;
    m_boundary_margin = boundary_margin;
}

bool Optimizer::IsNearBoundary(const cv::Point2f& pixel) const {
    if (!m_camera || m_boundary_margin <= 0) {
        return false;
    }
    return m_camera->IsNearBoundary(pixel, static_cast<float>(m_boundary_margin));
}

void Optimizer::PoseToParams(const Eigen::Matrix4f& pose, double* params) {
    // Convert SE3 matrix to tangent space [rho(3), phi(3)] using SE3::log()
    // This is consistent with SE3::exp() used in Factors
    SE3d se3(pose.cast<double>());
    Eigen::Matrix<double, 6, 1> xi = se3.log();
    
    // xi = [rho, phi] where rho is the translation part in tangent space
    params[0] = xi(0);
    params[1] = xi(1);
    params[2] = xi(2);
    params[3] = xi(3);
    params[4] = xi(4);
    params[5] = xi(5);
}

Eigen::Matrix4f Optimizer::ParamsToPose(const double* params) {
    // Convert tangent space [rho(3), phi(3)] to SE3 matrix using SE3::exp()
    // This is consistent with SE3::exp() used in Factors
    Eigen::Matrix<double, 6, 1> xi;
    xi << params[0], params[1], params[2], params[3], params[4], params[5];
    
    SE3d se3 = SE3d::exp(xi);
    return se3.matrix().cast<float>();
}

ceres::Solver::Options Optimizer::SetupSolverOptions(int max_iterations) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = max_iterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    return options;
}

PnPResult Optimizer::SolvePnP(std::shared_ptr<Frame> frame, bool fix_mappoints) {
    PnPResult result;
    
    if (!frame) {
        LOG_WARN("SolvePnP: null frame");
        return result;
    }
    
    // Collect observations with valid MapPoints, excluding boundary features
    std::vector<std::tuple<cv::Point2f, std::shared_ptr<MapPoint>, size_t>> observations;
    
    int total_features = 0;
    int invalid_features = 0;
    int no_mappoint = 0;
    int boundary_filtered = 0;
    
    const auto& features = frame->GetFeatures();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        total_features++;
        
        if (!feature || !feature->IsValid()) {
            invalid_features++;
            continue;
        }
        
        auto mp = frame->GetMapPoint(static_cast<int>(i));
        if (!mp || mp->IsBad()) {
            no_mappoint++;
            continue;
        }
        
        // Skip features near horizontal boundary (ERP wrap-around issue)
        if (IsNearBoundary(feature->GetPixelCoord())) {
            boundary_filtered++;
            continue;
        }
        
        observations.push_back({feature->GetPixelCoord(), mp, i});
    }
    
    LOG_DEBUG("  PnP stats: total={}, invalid={}, no_mp={}, boundary={}, valid={}",
              total_features, invalid_features, no_mappoint, boundary_filtered, observations.size());
    
    if (observations.size() < 6) {
        LOG_WARN("SolvePnP: insufficient observations ({})", observations.size());
        return result;
    }
    
    // Equirectangular camera parameters (cols, rows)
    double cols = static_cast<double>(frame->GetWidth());
    double rows = static_cast<double>(frame->GetHeight());
    factor::CameraParameters cam_params(cols, rows);
    
    // Get body-to-camera transform
    Eigen::Matrix4d T_cb = frame->GetTCB().cast<double>();
    
    // Information matrix
    Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
    
    // Store initial pose (for right perturbation approach)
    Eigen::Matrix4f T_wb_init = frame->GetTwb();
    Eigen::Matrix4d T_wb_init_d = T_wb_init.cast<double>();
    
    // Log initial pose for debugging
    Eigen::Vector3f init_pos = T_wb_init.block<3,1>(0,3);
    LOG_DEBUG("  [PnP] Frame {} init pose: ({:.4f},{:.4f},{:.4f}), {} observations",
              frame->GetFrameId(), init_pos.x(), init_pos.y(), init_pos.z(), observations.size());
    
    // Setup pose parameters as perturbation (initialized to zero)
    // Right perturbation: T_wb = T_wb_init * exp(delta_xi)
    double pose_params[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::PnPFactor*> factors;
    std::vector<size_t> feature_indices;
    std::vector<std::shared_ptr<MapPoint>> mappoints;  // Store MapPoints for marginalized check
    
    for (const auto& [pixel_coord, mp, feat_idx] : observations) {
        Eigen::Vector2d obs(pixel_coord.x, pixel_coord.y);
        Eigen::Vector3d world_pt = mp->GetPosition().cast<double>();
        
        // Pass initial pose to factor for right perturbation
        auto* cost = new factor::PnPFactor(obs, world_pt, cam_params, T_cb, T_wb_init_d, info);
        factors.push_back(cost);
        feature_indices.push_back(feat_idx);
        mappoints.push_back(mp);
        
        problem.AddResidualBlock(
            cost,
            new ceres::HuberLoss(m_huber_delta),
            pose_params
        );
    }
    
    // 4-round outlier detection
    const int num_rounds = 4;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int total_iterations = 0;
    
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    
    for (int round = 0; round < num_rounds; ++round) {
        // Reset perturbation to zero for each round (right perturbation approach)
        if (round > 0) {
            std::fill(pose_params, pose_params + 6, 0.0);
        }
        
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        if (round == 0) {
            initial_cost = summary.initial_cost;
        }
        final_cost = summary.final_cost;
        total_iterations += summary.iterations.size();
        
        // Bearing-based outlier detection for equirectangular cameras
        // Threshold: 2 degrees = 0.035 radians (more lenient than Stella's 1 degree)
        const double bearing_threshold = 2.0 * M_PI / 180.0;  // 2 degrees in radians
        
        const double* params_ptr = pose_params;
        int num_inliers = 0;
        int num_outliers = 0;
        double inlier_reproj_sum = 0.0;
        double max_chi2 = 0.0;
        double max_inlier_chi2 = 0.0;
        int chi2_bins[6] = {0};  // [0-2], [2-6], [6-10], [10-50], [50-100], [100+]
        
        for (size_t i = 0; i < factors.size(); ++i) {
            double chi2 = factors[i]->compute_chi_square(&params_ptr);
            
            // Marginalized MapPoints should never be marked as outliers (they preserve scale)
            bool is_marginalized = mappoints[i]->IsMarginalized();
            bool is_outlier = !is_marginalized && (chi2 > 5.991);
            
            // Track max chi2
            if (chi2 > max_chi2) max_chi2 = chi2;
            
            // Bin distribution
            if (chi2 < 2.0) chi2_bins[0]++;
            else if (chi2 < 6.0) chi2_bins[1]++;
            else if (chi2 < 10.0) chi2_bins[2]++;
            else if (chi2 < 50.0) chi2_bins[3]++;
            else if (chi2 < 100.0) chi2_bins[4]++;
            else chi2_bins[5]++;
            
            factors[i]->set_outlier(is_outlier);
            
            if (is_outlier) {
                num_outliers++;
            } else {
                num_inliers++;
                // Convert bearing error to approximate pixel error for logging
                // (bearing_error * image_width / (2*pi) gives rough pixel equivalent)
                inlier_reproj_sum += chi2;  // degrees for logging
                if (chi2 > max_inlier_chi2) max_inlier_chi2 = chi2;
            }
        }
        
        // Debug log for error distribution on final round
        if (round == num_rounds - 1) {
            LOG_DEBUG("  [PnP] chi2 dist: [0-2]={} [2-6]={} [6-10]={} [10-50]={} [50-100]={} [100+]={}",
                     chi2_bins[0], chi2_bins[1], chi2_bins[2], chi2_bins[3], chi2_bins[4], chi2_bins[5]);
            LOG_DEBUG("  [PnP] max_chi2={:.1f}, max_inlier_chi2={:.1f}, mean_inlier={:.2f}",
                     max_chi2, max_inlier_chi2, num_inliers > 0 ? inlier_reproj_sum / num_inliers : 0.0);
        }
        
        // Update result
        result.num_inliers = num_inliers;
        result.num_outliers = num_outliers;
        result.success = summary.IsSolutionUsable();
        
        // Compute mean inlier reprojection error
        if (num_inliers > 0) {
            final_cost = inlier_reproj_sum / num_inliers;
        }
        
        // Debug log for each round
        LOG_DEBUG("  PnP round {}: inliers={}, outliers={}, cost={:.2f}", 
                  round, num_inliers, num_outliers, final_cost);
    }
    
    // Compute final pose: T_wb = T_wb_init * exp(delta_xi) (right perturbation)
    Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params);
    SE3d T_wb_init_se3(T_wb_init_d);
    SE3d delta_T = SE3d::exp(delta_xi);
    SE3d T_wb_final = T_wb_init_se3 * delta_T;
    result.optimized_pose = T_wb_final.matrix().cast<float>();
    
    // Log pose change for debugging
    Eigen::Matrix4f old_pose = T_wb_init;
    Eigen::Vector3f old_pos = old_pose.block<3,1>(0,3);
    Eigen::Vector3f new_pos = result.optimized_pose.block<3,1>(0,3);
    float pos_change = (new_pos - old_pos).norm();
    
    // Safety check: reject if too few inliers or too large pose change
    const int min_inliers_for_update = 10;
    const float max_pose_change = 0.5f;  // 0.5m threshold
    
    if (result.num_inliers < min_inliers_for_update) {
        LOG_WARN("  PnP rejected: too few inliers ({} < {}), keeping predicted pose",
                 result.num_inliers, min_inliers_for_update);
        result.success = false;
        result.optimized_pose = old_pose;  // Keep old pose
    } else if (pos_change > max_pose_change && result.num_inliers < 50) {
        LOG_WARN("  PnP rejected: large pose change {:.3f}m with only {} inliers, keeping predicted pose",
                 pos_change, result.num_inliers);
        result.success = false;
        result.optimized_pose = old_pose;  // Keep old pose
    } else {
        if (pos_change > 0.3f) {
            LOG_WARN("  PnP large pose change: {:.3f}m, old=({:.2f},{:.2f},{:.2f}), new=({:.2f},{:.2f},{:.2f})",
                     pos_change, old_pos.x(), old_pos.y(), old_pos.z(),
                     new_pos.x(), new_pos.y(), new_pos.z());
        }
        frame->SetTwb(result.optimized_pose);
    }
    
    result.initial_cost = initial_cost;
    result.final_cost = final_cost;  // Now this is mean inlier reprojection error in pixels
    result.num_iterations = total_iterations;
    
    return result;
}

BAResult Optimizer::RunBA(const std::vector<std::shared_ptr<Frame>>& frames,
                          bool fix_first_pose,
                          bool fix_last_pose) {
    BAResult result;
    
    if (frames.size() < 2) {
        LOG_WARN("RunBA: need at least 2 frames");
        return result;
    }
    
    // Collect all MapPoints observed by these frames
    std::set<std::shared_ptr<MapPoint>> mappoint_set;
    for (const auto& frame : frames) {
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mappoint_set.insert(mp);
            }
        }
    }
    
    std::vector<std::shared_ptr<MapPoint>> mappoints(mappoint_set.begin(), mappoint_set.end());
    
    if (mappoints.empty()) {
        LOG_WARN("RunBA: no valid MapPoints");
        return result;
    }
    
    // Parameter blocks
    std::vector<std::array<double, 6>> pose_params(frames.size());
    std::vector<Eigen::Matrix4d> T_wb_inits(frames.size());  // Store initial poses for right perturbation
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize pose parameters as zero perturbation (right perturbation approach)
    for (size_t i = 0; i < frames.size(); ++i) {
        T_wb_inits[i] = frames[i]->GetTwb().cast<double>();
        pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Zero perturbation
    }
    
    // Initialize point parameters and create index map
    std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        Eigen::Vector3f pos = mappoints[i]->GetPosition();
        point_params[i] = {pos.x(), pos.y(), pos.z()};
        mp_to_idx[mappoints[i]] = i;
    }
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::BAFactor*> factors;
    std::vector<std::pair<size_t, size_t>> factor_indices;  // (frame_idx, mp_idx)
    
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const auto& frame = frames[fi];
        
        // Equirectangular camera parameters (cols, rows)
        double cols = static_cast<double>(frame->GetWidth());
        double rows = static_cast<double>(frame->GetHeight());
        factor::CameraParameters cam_params(cols, rows);
        
        Eigen::Matrix4d T_cb = frame->GetTCB().cast<double>();
        Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
        
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (!mp || mp->IsBad()) continue;
            
            // Skip features near horizontal boundary (ERP wrap-around issue)
            if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
            auto it = mp_to_idx.find(mp);
            if (it == mp_to_idx.end()) continue;
            
            size_t pi = it->second;
            Eigen::Vector2d obs(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
            // Pass initial pose for right perturbation
            auto* cost = new factor::BAFactor(obs, cam_params, T_cb, T_wb_inits[fi], info);
            factors.push_back(cost);
            factor_indices.push_back({fi, pi});
            
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(m_huber_delta),
                pose_params[fi].data(),
                point_params[pi].data()
            );
        }
    }
    
    // Fix poses as requested
    if (fix_first_pose && !frames.empty()) {
        problem.SetParameterBlockConstant(pose_params[0].data());
    }
    if (fix_last_pose && frames.size() > 1) {
        problem.SetParameterBlockConstant(pose_params.back().data());
    }
    
    // Solve
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Bearing-based outlier detection for equirectangular cameras
    const double bearing_threshold = 2.0 * M_PI / 180.0;  // 2 degrees
    
    int num_inliers = 0;
    int num_outliers = 0;
    
    // Count MapPoint outliers using bearing angle error
    std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
    std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
    for (size_t i = 0; i < factors.size(); ++i) {
        size_t fi = factor_indices[i].first;
        size_t pi = factor_indices[i].second;
        
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = factors[i]->compute_chi_square(params);
        
        bool is_outlier = (chi2 > 5.991);
        factors[i]->set_outlier(is_outlier);
        
        if (is_outlier) {
            num_outliers++;
            mp_outlier_count[mappoints[pi]]++;
        } else {
            num_inliers++;
            mp_inlier_count[mappoints[pi]]++;
        }
    }
    
    // Mark MapPoints as bad if ALL observations are outliers (more conservative)
    // But NEVER mark marginalized MapPoints as bad (they preserve scale)
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        if (mp->IsMarginalized()) continue;  // Marginalized MapPoints must not be removed
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        // Only mark bad if no inliers at all and at least 2 outlier observations
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    
    // Update frames and MapPoints
    // Compute final pose: T_wb = T_wb_init * exp(delta_xi) (right perturbation)
    for (size_t i = 0; i < frames.size(); ++i) {
        Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
        SE3d T_wb_init_se3(T_wb_inits[i]);
        SE3d delta_T = SE3d::exp(delta_xi);
        SE3d T_wb_final = T_wb_init_se3 * delta_T;
        frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
    }
    
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsBad() && !mappoints[i]->IsMarginalized()) {
            Eigen::Vector3f pos(point_params[i][0], point_params[i][1], point_params[i][2]);
            mappoints[i]->SetPosition(pos);
        }
    }
    
    result.success = summary.IsSolutionUsable();
    result.num_inliers = num_inliers;
    result.num_outliers = num_outliers;
    result.num_poses_optimized = frames.size();
    result.num_points_optimized = mappoints.size();
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();
    
    return result;
}

BAResult Optimizer::RunFullBA(const std::vector<std::shared_ptr<Frame>>& frames) {
    // Full BA: fix only first pose, optimize second pose and all MapPoints
    return RunBA(frames, true, false);
}

BAResult Optimizer::RunLocalBA(const std::vector<std::shared_ptr<Frame>>& window_frames) {
    BAResult result;
    
    if (window_frames.size() < 2) {
        LOG_WARN("RunLocalBA: need at least 2 frames in window");
        return result;
    }
    
    // Build set of window frame IDs for quick lookup
    std::set<int> window_frame_ids;
    for (const auto& frame : window_frames) {
        window_frame_ids.insert(frame->GetFrameId());
    }
    
    // Step 1: Collect all MapPoints observed by window frames
    std::set<std::shared_ptr<MapPoint>> mappoint_set;
    for (const auto& frame : window_frames) {
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mappoint_set.insert(mp);
            }
        }
    }
    
    std::vector<std::shared_ptr<MapPoint>> mappoints(mappoint_set.begin(), mappoint_set.end());
    
    if (mappoints.empty()) {
        LOG_WARN("RunLocalBA: no valid MapPoints");
        return result;
    }
    
    // Step 2: Only use window frames for BA (ignore out-of-window observations)
    std::vector<std::shared_ptr<Frame>> all_frames(window_frames.begin(), window_frames.end());
    
    // Create frame index map
    std::map<std::shared_ptr<Frame>, size_t> frame_to_idx;
    for (size_t i = 0; i < all_frames.size(); ++i) {
        frame_to_idx[all_frames[i]] = i;
    }
    
    // Parameter blocks
    std::vector<std::array<double, 6>> pose_params(all_frames.size());
    std::vector<Eigen::Matrix4d> T_wb_inits(all_frames.size());  // Store initial poses for right perturbation
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize pose parameters as zero perturbation (right perturbation approach)
    for (size_t i = 0; i < all_frames.size(); ++i) {
        T_wb_inits[i] = all_frames[i]->GetTwb().cast<double>();
        pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Zero perturbation
    }
    
    // Initialize point parameters and create index map
    std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        Eigen::Vector3f pos = mappoints[i]->GetPosition();
        point_params[i] = {pos.x(), pos.y(), pos.z()};
        mp_to_idx[mappoints[i]] = i;
    }
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::BAFactor*> factors;
    std::vector<std::pair<size_t, size_t>> factor_indices;  // (frame_idx, mp_idx)
    
    // Add residuals for ALL observations of ALL mappoints
    for (size_t pi = 0; pi < mappoints.size(); ++pi) {
        const auto& mp = mappoints[pi];
        const auto& observations = mp->GetObservations();
        
        for (const auto& obs : observations) {
            auto obs_frame = obs.frame.lock();
            if (!obs_frame) continue;
            
            auto frame_it = frame_to_idx.find(obs_frame);
            if (frame_it == frame_to_idx.end()) continue;
            
            size_t fi = frame_it->second;
            
            // Get the feature from the observation
            const auto& features = obs_frame->GetFeatures();
            if (obs.feature_index < 0 || obs.feature_index >= static_cast<int>(features.size())) continue;
            
            const auto& feature = features[obs.feature_index];
            if (!feature || !feature->IsValid()) continue;
            
            // Skip features near horizontal boundary (ERP wrap-around issue)
            if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
            // Camera parameters
            double cols = static_cast<double>(obs_frame->GetWidth());
            double rows = static_cast<double>(obs_frame->GetHeight());
            factor::CameraParameters cam_params(cols, rows);
            
            Eigen::Matrix4d T_cb = obs_frame->GetTCB().cast<double>();
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
            
            Eigen::Vector2d obs_pixel(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
            // Pass initial pose for right perturbation
            auto* cost = new factor::BAFactor(obs_pixel, cam_params, T_cb, T_wb_inits[fi], info);
            factors.push_back(cost);
            factor_indices.push_back({fi, pi});
            
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(m_huber_delta),
                pose_params[fi].data(),
                point_params[pi].data()
            );
        }
    }
    
    // Track which pose parameter blocks were actually added to the problem
    std::set<size_t> poses_in_problem;
    for (const auto& [fi, pi] : factor_indices) {
        poses_in_problem.insert(fi);
    }
    
    // For scale stability: fix only the first keyframe
    // All other keyframes are optimized
    constexpr size_t NUM_FIXED_KEYFRAMES = 1;
    int fixed_count = 0;
    int optimized_count = 0;
    
    for (size_t i = 0; i < all_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;
        
        if (i < NUM_FIXED_KEYFRAMES) {
            // Fix oldest N keyframes for scale stability
            problem.SetParameterBlockConstant(pose_params[i].data());
            fixed_count++;
        } else {
            optimized_count++;
        }
    }
    
    // Fix MapPoints that are marked as fixed (initialization points define scale)
    // Optimize MapPoints that are not fixed
    int fixed_mp_count = 0;
    int optimized_mp_count = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (mappoints[i]->IsMarginalized()) {
            problem.SetParameterBlockConstant(point_params[i].data());
            fixed_mp_count++;
        } else {
            optimized_mp_count++;
        }
    }
    
    LOG_DEBUG("  LocalBA: {} window frames (fix oldest {}, optimize {}), {} MapPoints ({} fixed, {} optimize), {} factors",
             window_frames.size(), fixed_count, optimized_count, mappoints.size(), fixed_mp_count, optimized_mp_count, factors.size());
    
    // Solve
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Chi-square based outlier detection (Stella VSLAM style)
    // chi_sq_2D = 5.99146 for 2 DOF with 5% significance level
    constexpr double chi_sq_threshold = 5.99146;
    
    int num_inliers = 0;
    int num_outliers = 0;
    
    std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
    std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
    for (size_t i = 0; i < factors.size(); ++i) {
        size_t fi = factor_indices[i].first;
        size_t pi = factor_indices[i].second;
        
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = factors[i]->compute_chi_square(params);
        
        bool is_outlier = (chi2 > chi_sq_threshold);
        factors[i]->set_outlier(is_outlier);
        
        if (is_outlier) {
            num_outliers++;
            mp_outlier_count[mappoints[pi]]++;
        } else {
            num_inliers++;
            mp_inlier_count[mappoints[pi]]++;
        }
    }
    
    // Mark MapPoints as bad if ALL observations are outliers
    // But NEVER mark marginalized MapPoints as bad (they preserve scale)
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        if (mp->IsMarginalized()) continue;  // Marginalized MapPoints must not be removed
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    if (bad_mp_count > 0) {
        LOG_DEBUG("  LocalBA: marked {} MapPoints as bad", bad_mp_count);
    }
    
    // Update all optimized frame poses (frames after NUM_FIXED_KEYFRAMES)
    // Compute final pose: T_wb = T_wb_init * exp(delta_xi) (right perturbation)
    for (size_t i = NUM_FIXED_KEYFRAMES; i < all_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;  // Skip if not in problem
        
        Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
        SE3d T_wb_init_se3(T_wb_inits[i]);
        SE3d delta_T = SE3d::exp(delta_xi);
        SE3d T_wb_final = T_wb_init_se3 * delta_T;
        all_frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
    }
    
    // Update MapPoints that were optimized (not fixed)
    int mp_updated = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsMarginalized()) {
            Eigen::Vector3f new_pos(point_params[i][0], point_params[i][1], point_params[i][2]);
            mappoints[i]->SetPosition(new_pos);
            mp_updated++;
        }
    }
    
    result.success = summary.IsSolutionUsable();
    result.num_inliers = num_inliers;
    result.num_outliers = num_outliers;
    result.num_poses_optimized = window_frames.size() - 1 + fixed_count;  // -1 for fixed first frame
    result.num_points_optimized = mappoints.size();
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();
    
    return result;
}

} // namespace vio_360
