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
#include "util/Logger.h"
#include "util/LieUtils.h"

#include <set>

namespace vio_360 {

Optimizer::Optimizer()
    : m_huber_delta(1.0)
    , m_pixel_noise_std(1.0)
    , m_max_iterations(50)
    , m_chi2_threshold(9.21) {  // Chi-square 99% for 2 DOF (more permissive)
}

void Optimizer::PoseToParams(const Eigen::Matrix4f& pose, double* params) {
    // Convert SE3 matrix to tangent space [rot(3), trans(3)]
    Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
    Eigen::Vector3f t = pose.block<3, 1>(0, 3);
    
    // Rotation to axis-angle using Rodrigues
    Eigen::AngleAxisf aa(R);
    Eigen::Vector3f axis_angle = aa.axis() * aa.angle();
    
    // [rotation, translation]
    params[0] = axis_angle.x();
    params[1] = axis_angle.y();
    params[2] = axis_angle.z();
    params[3] = t.x();
    params[4] = t.y();
    params[5] = t.z();
}

Eigen::Matrix4f Optimizer::ParamsToPose(const double* params) {
    // Convert tangent space [rot(3), trans(3)] to SE3 matrix
    Eigen::Vector3f axis_angle(params[0], params[1], params[2]);
    Eigen::Vector3f t(params[3], params[4], params[5]);
    
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
    
    // Collect observations with valid MapPoints
    std::vector<std::tuple<cv::Point2f, std::shared_ptr<MapPoint>, size_t>> observations;
    
    const auto& features = frame->GetFeatures();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (!feature || !feature->IsValid()) continue;
        
        auto mp = frame->GetMapPoint(static_cast<int>(i));
        if (!mp || mp->IsBad()) continue;
        
        observations.push_back({feature->GetPixelCoord(), mp, i});
    }
    
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
        
        // Outlier detection using chi-square test
        const double* params_ptr = pose_params;
        int num_inliers = 0;
        int num_outliers = 0;
        
        for (size_t i = 0; i < factors.size(); ++i) {
            double chi2 = factors[i]->compute_chi_square(&params_ptr);
            bool is_outlier = (chi2 > m_chi2_threshold);
            
            factors[i]->set_outlier(is_outlier);
            
            // Mark feature as invalid if outlier
            auto& feature = frame->GetFeatures()[feature_indices[i]];
            if (feature) {
                feature->SetValid(!is_outlier);
            }
            
            if (is_outlier) {
                num_outliers++;
            } else {
                num_inliers++;
            }
        }
        
        // Update result
        result.num_inliers = num_inliers;
        result.num_outliers = num_outliers;
        result.success = summary.IsSolutionUsable();
    }
    
    // Update frame pose with final optimized value
    result.optimized_pose = ParamsToPose(pose_params);
    frame->SetTwb(result.optimized_pose);
    
    result.initial_cost = initial_cost;
    result.final_cost = final_cost;
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
    
    // Outlier detection and MapPoint quality assessment
    int num_inliers = 0;
    int num_outliers = 0;
    
    // Count inliers/outliers per MapPoint
    std::map<size_t, int> mp_inlier_count;
    std::map<size_t, int> mp_outlier_count;
    
    for (size_t i = 0; i < factors.size(); ++i) {
        auto [fi, pi] = factor_indices[i];
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = factors[i]->compute_chi_square(params);
        if (chi2 > m_chi2_threshold) {
            num_outliers++;
            mp_outlier_count[pi]++;
        } else {
            num_inliers++;
            mp_inlier_count[pi]++;
        }
    }
    
    // Mark MapPoints as bad if mostly outliers
    int bad_mp_count = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        int inliers = mp_inlier_count[i];
        int outliers = mp_outlier_count[i];
        int total = inliers + outliers;
        
        // Mark as bad only if:
        // 1. No inliers at all AND has outliers, OR
        // 2. Outliers are more than 2x inliers
        if (total > 0 && (inliers == 0 || outliers > 2 * inliers)) {
            mappoints[i]->SetBad(true);
            bad_mp_count++;
        }
    }
    
    if (bad_mp_count > 0) {
        LOG_INFO("  Marked {} MapPoints as bad", bad_mp_count);
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

} // namespace vio_360
