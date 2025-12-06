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
#include "processing/IMUPreintegrator.h"
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
        // Threshold: 2 degrees = 0.035 radians
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
    } 
     
    else {
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

BAResult Optimizer::RunVIBA(const std::vector<std::shared_ptr<Frame>>& frames,
                            const Eigen::Vector3f& gravity,
                            bool fix_first_pose) {
    BAResult result;
    
    if (frames.size() < 2) {
        LOG_WARN("RunVIBA: need at least 2 frames");
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
        LOG_WARN("RunVIBA: no valid MapPoints");
        return result;
    }
    
    // ==================== Parameter blocks ====================
    // Pose parameters (6D tangent space, right perturbation)
    std::vector<std::array<double, 6>> pose_params(frames.size());
    std::vector<Eigen::Matrix4d> T_wb_inits(frames.size());
    
    // Velocity parameters (3D each frame)
    std::vector<std::array<double, 3>> velocity_params(frames.size());
    
    // Shared bias parameters
    std::array<double, 3> gyro_bias_params;
    std::array<double, 3> accel_bias_params;
    
    // Point parameters
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize parameters
    for (size_t i = 0; i < frames.size(); ++i) {
        T_wb_inits[i] = frames[i]->GetTwb().cast<double>();
        pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Zero perturbation
        
        Eigen::Vector3f vel = frames[i]->GetVelocity();
        velocity_params[i] = {vel.x(), vel.y(), vel.z()};
    }
    
    // Initialize bias from first frame
    Eigen::Vector3f gb = frames[0]->GetGyroBias();
    Eigen::Vector3f ab = frames[0]->GetAccelBias();
    gyro_bias_params = {gb.x(), gb.y(), gb.z()};
    accel_bias_params = {ab.x(), ab.y(), ab.z()};
    
    // Initialize point parameters
    std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        Eigen::Vector3f pos = mappoints[i]->GetPosition();
        point_params[i] = {pos.x(), pos.y(), pos.z()};
        mp_to_idx[mappoints[i]] = i;
    }
    
    // ==================== Build optimization problem ====================
    ceres::Problem problem;
    std::vector<factor::BAFactor*> ba_factors;
    std::vector<std::pair<size_t, size_t>> ba_factor_indices;  // (frame_idx, mp_idx)
    
    // Add visual factors (BAFactor)
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const auto& frame = frames[fi];
        
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
            
            if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
            auto it = mp_to_idx.find(mp);
            if (it == mp_to_idx.end()) continue;
            
            size_t pi = it->second;
            Eigen::Vector2d obs(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
            auto* cost = new factor::BAFactor(obs, cam_params, T_cb, T_wb_inits[fi], info);
            ba_factors.push_back(cost);
            ba_factor_indices.push_back({fi, pi});
            
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(m_huber_delta),
                pose_params[fi].data(),
                point_params[pi].data()
            );
        }
    }
    
    // Add IMU factors (InertialFactor between consecutive keyframes)
    // Gravity direction is fixed (already aligned to -Z)
    Eigen::Vector3d g_world = gravity.cast<double>();
    
    for (size_t i = 1; i < frames.size(); ++i) {
        auto preint = frames[i]->GetIMUPreintegrationFromLastKeyframe();
        if (!preint) {
            LOG_WARN("RunVIBA: Frame {} missing IMU preintegration", frames[i]->GetFrameId());
            continue;
        }
        
        // Create fixed-gravity inertial factor
        auto* imu_cost = new factor::InertialFactorFixedGravity(
            preint, g_world, T_wb_inits[i-1], T_wb_inits[i]);
        
        problem.AddResidualBlock(
            imu_cost,
            nullptr,  // No robust loss for IMU
            pose_params[i-1].data(),    // pose_i
            velocity_params[i-1].data(), // velocity_i
            gyro_bias_params.data(),     // shared gyro bias
            accel_bias_params.data(),    // shared accel bias
            pose_params[i].data(),       // pose_j
            velocity_params[i].data()    // velocity_j
        );
    }
    
    // Fix first pose if requested
    if (fix_first_pose && !frames.empty()) {
        problem.SetParameterBlockConstant(pose_params[0].data());
    }
    
    // ==================== Solve ====================
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    LOG_INFO("VIBA: iterations={}, cost {:.4f} -> {:.4f}", 
             summary.iterations.size(), summary.initial_cost, summary.final_cost);
    
    // ==================== Outlier detection ====================
    int num_inliers = 0;
    int num_outliers = 0;
    
    std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
    std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
    for (size_t i = 0; i < ba_factors.size(); ++i) {
        size_t fi = ba_factor_indices[i].first;
        size_t pi = ba_factor_indices[i].second;
        
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = ba_factors[i]->compute_chi_square(params);
        
        bool is_outlier = (chi2 > 5.991);
        ba_factors[i]->set_outlier(is_outlier);
        
        if (is_outlier) {
            num_outliers++;
            mp_outlier_count[mappoints[pi]]++;
        } else {
            num_inliers++;
            mp_inlier_count[mappoints[pi]]++;
        }
    }
    
    // Mark bad MapPoints
    for (const auto& mp : mappoints) {
        if (mp->IsMarginalized()) continue;
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
        }
    }
    
    // ==================== Update results ====================
    // Update poses
    for (size_t i = 0; i < frames.size(); ++i) {
        Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
        SE3d T_wb_init_se3(T_wb_inits[i]);
        SE3d delta_T = SE3d::exp(delta_xi);
        SE3d T_wb_final = T_wb_init_se3 * delta_T;
        frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
        
        // Update velocity
        Eigen::Vector3f vel(velocity_params[i][0], velocity_params[i][1], velocity_params[i][2]);
        frames[i]->SetVelocity(vel);
    }
    
    // Update biases for all frames
    Eigen::Vector3f new_gyro_bias(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
    Eigen::Vector3f new_accel_bias(accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    for (auto& frame : frames) {
        frame->SetGyroBias(new_gyro_bias);
        frame->SetAccelBias(new_accel_bias);
    }
    
    // Update MapPoints
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
    
    // Chi-square based outlier detection
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

// ============================================================================
// IMU Initialization Optimization
// ============================================================================

IMUInitResult Optimizer::OptimizeIMUInit(
    const std::vector<std::shared_ptr<Frame>>& frames) {
    
    IMUInitResult result;
    
    // Need at least 3 keyframes for IMU initialization
    if (frames.size() < 3) {
        LOG_WARN("[IMU_INIT] Need at least 3 keyframes, got {}", frames.size());
        return result;
    }
    
    // Check all frames have preintegration (except first)
    for (size_t i = 1; i < frames.size(); ++i) {
        if (!frames[i]->HasIMUPreintegrationFromLastKeyframe()) {
            LOG_WARN("[IMU_INIT] Frame {} missing preintegration", frames[i]->GetFrameId());
            return result;
        }
    }
    
    LOG_INFO("========================================================");
    LOG_INFO("[IMU_INIT] Starting 2-Stage Optimization");
    LOG_INFO("  Keyframes: {}", frames.size());
    LOG_INFO("  Mode: 360 Monocular (gravity + scale optimization)");
    LOG_INFO("========================================================");
    
    size_t num_frames = frames.size();
    
    // =========================================================================
    // SETUP: Initialize parameter vectors
    // =========================================================================
    
    // Pose parameters (tangent space): one per frame
    std::vector<std::vector<double>> pose_params(num_frames, std::vector<double>(6, 0.0));
    std::vector<Eigen::Matrix4d> T_wb_inits(num_frames);
    
    // Velocity parameters: one per frame
    std::vector<std::vector<double>> velocity_params(num_frames, std::vector<double>(3, 0.0));
    
    // Bias parameters: shared across all frames
    std::vector<double> gyro_bias_params(3, 0.0);
    std::vector<double> accel_bias_params(3, 0.0);
    
    // Gravity direction: 2D parameterization (theta_x, theta_y)
    std::vector<double> gravity_dir_params(2, 0.0);
    
    // Scale: fixed at 1.0 for now
    std::vector<double> scale_params(1, 1.0);
    
    // Initialize parameters from frames
    for (size_t i = 0; i < num_frames; ++i) {
        // Store initial pose for right perturbation
        T_wb_inits[i] = frames[i]->GetTwb().cast<double>();
        
        // Pose starts at zero (identity perturbation)
        for (int j = 0; j < 6; ++j) pose_params[i][j] = 0.0;
        
        // Initialize velocity from preintegration (like reference code)
        // velocity_world = R_wb_prev * delta_V
        if (i > 0) {
            auto preint = frames[i]->GetIMUPreintegrationFromLastKeyframe();
            if (preint && preint->dt_total > 0.001) {
                // Use previous frame's rotation to transform delta_V to world frame
                Eigen::Matrix3d R_wb_prev = T_wb_inits[i-1].block<3,3>(0,0);
                Eigen::Vector3d vel_world = R_wb_prev * preint->delta_V.cast<double>();
                for (int j = 0; j < 3; ++j) velocity_params[i][j] = vel_world(j);
                LOG_DEBUG("  Frame {}: velocity from preint = [{:.3f}, {:.3f}, {:.3f}]",
                          frames[i]->GetFrameId(), vel_world(0), vel_world(1), vel_world(2));
            }
        }
    }
    
    // Initialize gravity direction from first frame's orientation
    // Assume initial world frame has gravity pointing down (-Z)
    // g_world = R_wg * [0, 0, -9.81]^T
    // Initial guess: gravity_dir = [0, 0] means g = [0, 0, -9.81]
    gravity_dir_params[0] = 0.0;
    gravity_dir_params[1] = 0.0;
    
    LOG_INFO("[SETUP] Initial parameters:");
    LOG_INFO("  Gravity direction: [{:.4f}, {:.4f}]", gravity_dir_params[0], gravity_dir_params[1]);
    LOG_INFO("  Scale: {:.4f} (will be optimized)", scale_params[0]);
    
    // =========================================================================
    // STAGE 1: Optimize Gravity Direction + Scale
    // =========================================================================
    
    LOG_INFO("");
    LOG_INFO("╔══════════════════════════════════════════════════════╗");
    LOG_INFO("║ STAGE 1: Gravity + Scale Optimization               ║");
    LOG_INFO("╚══════════════════════════════════════════════════════╝");
    
    {
        ceres::Problem problem;
        
        // Add InertialGravityScaleFactor between consecutive frames
        int factors_added = 0;
        for (size_t i = 0; i < num_frames - 1; ++i) {
            auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
            if (!preint) continue;
            
            double dt = preint->dt_total;
            if (dt < 0.001 || dt > 2.0) {
                LOG_WARN("  Invalid dt={:.4f}s between frames {} and {}", 
                         dt, frames[i]->GetFrameId(), frames[i+1]->GetFrameId());
                continue;
            }
            
            auto* factor = new factor::InertialGravityScaleFactor(preint, 9.81);
            
            // Huber loss for robustness
            auto* loss = new ceres::HuberLoss(std::sqrt(16.0));
            
            problem.AddResidualBlock(factor, loss,
                pose_params[i].data(),
                velocity_params[i].data(),
                gyro_bias_params.data(),
                accel_bias_params.data(),
                pose_params[i + 1].data(),
                velocity_params[i + 1].data(),
                gravity_dir_params.data(),
                scale_params.data());
            
            factors_added++;
        }
        
        if (factors_added == 0) {
            LOG_WARN("[IMU_INIT] No valid factors added");
            return result;
        }
        
        LOG_INFO("  Added {} InertialGravityScaleFactor(s)", factors_added);
        
        // Fix poses, velocities, biases in Stage 1
        // Only gravity_dir and scale are FREE (for monocular)
        for (size_t i = 0; i < num_frames; ++i) {
            problem.SetParameterBlockConstant(pose_params[i].data());
            problem.SetParameterBlockConstant(velocity_params[i].data());
        }
        problem.SetParameterBlockConstant(gyro_bias_params.data());
        problem.SetParameterBlockConstant(accel_bias_params.data());
        
        // Scale is FREE (always optimize for 360 monocular)
        // gravity_dir and scale are both FREE
        
        // Solve
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = false;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        result.initial_cost = summary.initial_cost;
        
        LOG_INFO("  Stage 1 result:");
        LOG_INFO("    Iterations: {}", summary.iterations.size());
        LOG_INFO("    Cost: {:.6f} -> {:.6f}", summary.initial_cost, summary.final_cost);
        LOG_INFO("    Gravity dir: [{:.4f}, {:.4f}]", gravity_dir_params[0], gravity_dir_params[1]);
        LOG_INFO("    Scale: {:.4f} (optimized)", scale_params[0]);
    }
    
    // =========================================================================
    // STAGE 2: Optimize Velocities and Biases (gravity and scale fixed)
    // =========================================================================
    
    LOG_INFO("");
    LOG_INFO("╔══════════════════════════════════════════════════════╗");
    LOG_INFO("║ STAGE 2: Velocity + Bias Optimization                ║");
    LOG_INFO("╚══════════════════════════════════════════════════════╝");
    
    {
        ceres::Problem problem;
        
        // Add InertialGravityScaleFactor between consecutive frames
        int factors_added = 0;
        for (size_t i = 0; i < num_frames - 1; ++i) {
            auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
            if (!preint) continue;
            
            double dt = preint->dt_total;
            if (dt < 0.001 || dt > 2.0) continue;
            
            auto* factor = new factor::InertialGravityScaleFactor(preint, 9.81);
            auto* loss = new ceres::HuberLoss(std::sqrt(16.0));
            
            problem.AddResidualBlock(factor, loss,
                pose_params[i].data(),
                velocity_params[i].data(),
                gyro_bias_params.data(),
                accel_bias_params.data(),
                pose_params[i + 1].data(),
                velocity_params[i + 1].data(),
                gravity_dir_params.data(),
                scale_params.data());
            
            factors_added++;
        }
        
        // Add bias prior (regularization)
        Eigen::Vector3d zero_bias = Eigen::Vector3d::Zero();
        double bias_weight = 1.0;  // Weak prior
        
        auto* gyro_prior = new factor::BiasPriorFactor(zero_bias, bias_weight);
        auto* accel_prior = new factor::BiasPriorFactor(zero_bias, bias_weight);
        
        problem.AddResidualBlock(gyro_prior, nullptr, gyro_bias_params.data());
        problem.AddResidualBlock(accel_prior, nullptr, accel_bias_params.data());
        
        // Fix poses, gravity_dir, and scale in Stage 2
        for (size_t i = 0; i < num_frames; ++i) {
            problem.SetParameterBlockConstant(pose_params[i].data());
        }
        problem.SetParameterBlockConstant(gravity_dir_params.data());
        problem.SetParameterBlockConstant(scale_params.data());
        
        // Velocities and biases are FREE
        
        // Solve
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = false;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        result.final_cost = summary.final_cost;
        
        LOG_INFO("  Stage 2 result:");
        LOG_INFO("    Iterations: {}", summary.iterations.size());
        LOG_INFO("    Cost: {:.6f} -> {:.6f}", summary.initial_cost, summary.final_cost);
        LOG_INFO("    Gyro bias: [{:.6f}, {:.6f}, {:.6f}]", 
                 gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
        LOG_INFO("    Accel bias: [{:.6f}, {:.6f}, {:.6f}]", 
                 accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    }
    
    // =========================================================================
    // Extract results
    // =========================================================================
    
    // Compute gravity vector from direction
    double theta_x = gravity_dir_params[0];
    double theta_y = gravity_dir_params[1];
    Eigen::Vector3d omega(theta_x, theta_y, 0.0);
    double angle = omega.norm();
    Eigen::Matrix3d R_wg;
    if (angle < 1e-6) {
        R_wg = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = omega / angle;
        R_wg = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }
    Eigen::Vector3d g_I(0, 0, -9.81);
    Eigen::Vector3d gravity = R_wg * g_I;
    
    result.success = true;
    result.gravity = gravity.cast<float>();
    result.Rwg = R_wg.cast<float>();  // Store Rwg for coordinate transformation
    result.scale = scale_params[0];
    result.gyro_bias = Eigen::Vector3f(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
    result.accel_bias = Eigen::Vector3f(accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    
    // Store velocities
    result.velocities.resize(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        result.velocities[i] = Eigen::Vector3f(
            velocity_params[i][0], velocity_params[i][1], velocity_params[i][2]);
    }
    
    LOG_INFO("");
    LOG_INFO("========================================================");
    LOG_INFO("[IMU_INIT] Optimization Complete!");
    LOG_INFO("  Gravity: [{:.4f}, {:.4f}, {:.4f}] (magnitude: {:.4f})",
             result.gravity.x(), result.gravity.y(), result.gravity.z(),
             result.gravity.norm());
    LOG_INFO("  Scale: {:.4f}", result.scale);
    LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}]",
             result.gyro_bias.x(), result.gyro_bias.y(), result.gyro_bias.z());
    LOG_INFO("  Accel bias: [{:.6f}, {:.6f}, {:.6f}]",
             result.accel_bias.x(), result.accel_bias.y(), result.accel_bias.z());
    LOG_INFO("========================================================");
    
    return result;
}

} // namespace vio_360
