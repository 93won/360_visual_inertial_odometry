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
    // Convert SE3 matrix to tangent space [trans(3), rot(3)]
    // Order matches Jacobian column order in PnPFactor and BAFactor
    Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
    Eigen::Vector3f t = pose.block<3, 1>(0, 3);
    
    // Rotation to axis-angle using Rodrigues
    Eigen::AngleAxisf aa(R);
    Eigen::Vector3f axis_angle = aa.axis() * aa.angle();
    
    // [translation, rotation] - matches Jacobian order
    params[0] = t.x();
    params[1] = t.y();
    params[2] = t.z();
    params[3] = axis_angle.x();
    params[4] = axis_angle.y();
    params[5] = axis_angle.z();
}

Eigen::Matrix4f Optimizer::ParamsToPose(const double* params) {
    // Convert tangent space [trans(3), rot(3)] to SE3 matrix
    // Order matches PoseToParams
    Eigen::Vector3f t(params[0], params[1], params[2]);
    Eigen::Vector3f axis_angle(params[3], params[4], params[5]);
    
    float angle = axis_angle.norm();
    Eigen::Matrix3f R;
    if (angle < 1e-8f) {
        R = Eigen::Matrix3f::Identity();
    } else {
        Eigen::Vector3f axis = axis_angle / angle;
        R = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
    }
    
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    return pose;
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
    
    // Store initial pose for reset each round
    double initial_pose_params[6];
    PoseToParams(frame->GetTwb(), initial_pose_params);
    
    // Setup pose parameters (SE3 tangent space)
    double pose_params[6];
    std::copy(initial_pose_params, initial_pose_params + 6, pose_params);
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::PnPFactor*> factors;
    std::vector<size_t> feature_indices;
    
    for (const auto& [pixel_coord, mp, feat_idx] : observations) {
        Eigen::Vector2d obs(pixel_coord.x, pixel_coord.y);
        Eigen::Vector3d world_pt = mp->GetPosition().cast<double>();
        
        auto* cost = new factor::PnPFactor(obs, world_pt, cam_params, T_cb, info);
        factors.push_back(cost);
        feature_indices.push_back(feat_idx);
        
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
        // Reset pose to initial value for each round
        if (round > 0) {
            std::copy(initial_pose_params, initial_pose_params + 6, pose_params);
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
        
        for (size_t i = 0; i < factors.size(); ++i) {
            double chi2 = factors[i]->compute_chi_square(&params_ptr);
            bool is_outlier = (chi2 > 5.991);
            
            factors[i]->set_outlier(is_outlier);
            
            if (is_outlier) {
                num_outliers++;
            } else {
                num_inliers++;
                // Convert bearing error to approximate pixel error for logging
                // (bearing_error * image_width / (2*pi) gives rough pixel equivalent)
                inlier_reproj_sum += chi2;  // degrees for logging
            }
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
    
    // Update frame pose with final optimized value
    result.optimized_pose = ParamsToPose(pose_params);
    
    // Log pose change for debugging
    Eigen::Matrix4f old_pose = frame->GetTwb();
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
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize pose parameters
    for (size_t i = 0; i < frames.size(); ++i) {
        PoseToParams(frames[i]->GetTwb(), pose_params[i].data());
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
            
            auto* cost = new factor::BAFactor(obs, cam_params, T_cb, info);
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
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        // Only mark bad if no inliers at all and at least 2 outlier observations
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    
    // Update frames and MapPoints
    for (size_t i = 0; i < frames.size(); ++i) {
        frames[i]->SetTwb(ParamsToPose(pose_params[i].data()));
    }
    
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsBad()) {
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
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize pose parameters
    for (size_t i = 0; i < all_frames.size(); ++i) {
        PoseToParams(all_frames[i]->GetTwb(), pose_params[i].data());
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
            
            auto* cost = new factor::BAFactor(obs_pixel, cam_params, T_cb, info);
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
    
    // For scale stability: fix ALL keyframes except the last one
    // Only the last (newest) keyframe pose is optimized
    int fixed_count = 0;
    int optimized_count = 0;
    size_t last_frame_idx = all_frames.size() - 1;
    
    for (size_t i = 0; i < all_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;
        
        if (i != last_frame_idx) {
            // Fix all keyframes except the last one
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
        if (mappoints[i]->IsFixed()) {
            problem.SetParameterBlockConstant(point_params[i].data());
            fixed_mp_count++;
        } else {
            optimized_mp_count++;
        }
    }
    
    LOG_INFO("  LocalBA: {} window frames (fix {}, optimize last only), {} MapPoints ({} fixed, {} optimize), {} factors",
             window_frames.size(), fixed_count, mappoints.size(), fixed_mp_count, optimized_mp_count, factors.size());
    
    // Debug: print poses before BA
    LOG_INFO("  Before BA poses:");
    size_t last_idx = window_frames.size() - 1;
    for (size_t i = 0; i < window_frames.size(); ++i) {
        Eigen::Matrix4f T = window_frames[i]->GetTwb();
        Eigen::Vector3f pos = T.block<3,1>(0,3);
        LOG_INFO("    Frame {}: pos=({:.4f},{:.4f},{:.4f}){}", 
                 window_frames[i]->GetFrameId(), pos.x(), pos.y(), pos.z(),
                 i != last_idx ? " [FIXED]" : " [OPTIMIZE]");
    }
    
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
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    if (bad_mp_count > 0) {
        LOG_INFO("  LocalBA: marked {} MapPoints as bad", bad_mp_count);
    }
    
    // Update only the last keyframe pose (only one being optimized)
    size_t last_frame_idx_update = all_frames.size() - 1;
    all_frames[last_frame_idx_update]->SetTwb(ParamsToPose(pose_params[last_frame_idx_update].data()));
    
    // Debug: print poses after BA
    LOG_INFO("  After BA poses:");
    for (size_t i = 0; i < window_frames.size(); ++i) {
        Eigen::Matrix4f T = window_frames[i]->GetTwb();
        Eigen::Vector3f pos = T.block<3,1>(0,3);
        LOG_INFO("    Frame {}: pos=({:.4f},{:.4f},{:.4f}){}", 
                 window_frames[i]->GetFrameId(), pos.x(), pos.y(), pos.z(),
                 i != last_frame_idx_update ? " [FIXED]" : " [OPTIMIZED]");
    }
    
    // Update MapPoints that were optimized (not fixed)
    int mp_updated = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsFixed()) {
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
