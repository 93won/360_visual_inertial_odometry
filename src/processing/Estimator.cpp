/**
 * @file      Estimator.cpp
 * @brief     Main VIO estimator implementation
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "FeatureTracker.h"
#include "Initializer.h"
#include "IMUPreintegrator.h"
#include "Optimizer.h"
#include "Camera.h"
#include "Frame.h"
#include "Feature.h"
#include "MapPoint.h"
#include "ConfigUtils.h"
#include "Logger.h"

namespace vio_360 {

Estimator::Estimator()
    : m_frame_id_counter(0)
    , m_initialized(false)
    , m_current_pose(Eigen::Matrix4f::Identity()) {
    
    // Initialize camera
    const auto& config = ConfigUtils::GetInstance();
    m_camera = std::make_shared<Camera>(
        config.camera_width,
        config.camera_height
    );
    
    // Initialize feature tracker
    m_feature_tracker = std::make_unique<FeatureTracker>(
        m_camera,
        config.max_features,
        config.min_distance,
        config.quality_level
    );
    
    // Initialize monocular initializer
    m_initializer = std::make_unique<Initializer>();
    
    // Initialize IMU preintegrator
    m_imu_preintegrator = std::make_unique<IMUPreintegrator>();
    
    // Load initialization parameters from config
    m_window_size = config.initialization_window_size;
    m_min_parallax = config.initialization_min_parallax;
    m_frame_window.reserve(m_window_size);
}

Estimator::~Estimator() {
    // Cleanup if needed
}

Estimator::EstimationResult Estimator::ProcessFrame(const cv::Mat& image, double timestamp) {
    EstimationResult result;
    
    // Create new frame
    m_current_frame = CreateFrame(image, timestamp);
    
    if (!m_initialized) {
        // Not initialized yet - accumulate frames for initialization
        if (m_previous_frame) {
            // Track features for initialization
            result.num_tracked = TrackFeatures();
        } else {
            // First frame - detect features
            DetectFeatures();
        }
        
        // Add current frame to window
        m_frame_window.push_back(m_current_frame);
        
        // Maintain window size
        if (static_cast<int>(m_frame_window.size()) > m_window_size) {
            m_frame_window.erase(m_frame_window.begin());
        }
        
        // Try to initialize when window is full
        if (static_cast<int>(m_frame_window.size()) == m_window_size) {
            bool init_result = TryInitialize();
            
            if (init_result) {
                // Initialization succeeded!
                result.init_success = true;
                m_initialized = true;
                
                // Set initialization keyframes
                auto first_frame = m_frame_window.front();
                auto last_frame = m_frame_window.back();
                
                first_frame->SetKeyframe(true);
                last_frame->SetKeyframe(true);
                
                m_keyframes.push_back(first_frame);
                m_keyframes.push_back(last_frame);
                m_last_keyframe = last_frame;
                
                LOG_INFO("Set keyframes: {} and {} (last_keyframe={})", 
                         first_frame->GetFrameId(), last_frame->GetFrameId(), 
                         m_last_keyframe->GetFrameId());
            } else {
                // Check if ready for initialization (sufficient parallax)
                if (m_frame_window.size() >= 2) {
                    auto first_frame = m_frame_window.front();
                    auto last_frame = m_frame_window.back();
                    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
                    
                    if (parallax >= m_min_parallax) {
                        result.init_ready = true;
                        // Pause even on failure for debugging
                        result.init_success = true;  // Trigger pause for debugging
                    }
                }
            }
        }
    } else {
        // Already initialized - normal tracking
        
        // Track features from previous frame
        int num_tracked = TrackFeatures();
        
        // Link MapPoints from previous frame to current frame
        LinkMapPointsFromPreviousFrame();
        
        // Count valid MapPoints
        int valid_mp_count = 0;
        const auto& features = m_current_frame->GetFeatures();
        for (size_t j = 0; j < features.size(); ++j) {
            auto mp = m_current_frame->GetMapPoint(static_cast<int>(j));
            if (mp && !mp->IsBad()) {
                valid_mp_count++;
            }
        }
        
        // Pose estimation using PnP
        if (valid_mp_count >= 6) {
            // Initialize with previous pose as prior
            m_current_frame->SetTwb(m_previous_frame->GetTwb());
            
            // Run PnP optimization
            Optimizer optimizer;
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            // Compute parallax from last keyframe
            float parallax_from_kf = 0.0f;
            if (m_last_keyframe) {
                parallax_from_kf = ComputeParallax(m_last_keyframe, m_current_frame);
            }
            
            if (pnp_result.success) {
                m_current_pose = m_current_frame->GetTwb();
                LOG_INFO("Frame {}: PnP {} in/{} out, reproj {:.2f}px, parallax {:.1f}px",
                         m_current_frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                         pnp_result.final_cost, parallax_from_kf);
                result.success = true;
            } else {
                LOG_WARN("Frame {}: PnP failed", m_current_frame->GetFrameId());
                result.success = false;
            }
        } else {
            LOG_WARN("Frame {}: Not enough MapPoints ({}) for PnP", 
                     m_current_frame->GetFrameId(), valid_mp_count);
            result.success = false;
        }
        
        // Detect new features if needed
        if (m_current_frame->GetFeatureCount() < 100) {
            DetectFeatures();
        }
        
        // Check if should create keyframe
        if (ShouldCreateKeyframe()) {
            CreateKeyframe();
        }
        
        result.num_tracked = num_tracked;
    }
    
    // Update state
    result.pose = m_current_pose;
    result.num_features = m_current_frame->GetFeatureCount();
    m_all_frames.push_back(m_current_frame);
    m_previous_frame = m_current_frame;
    
    return result;
}

Estimator::EstimationResult Estimator::ProcessFrame(
    const cv::Mat& image, 
    double timestamp,
    const std::vector<IMUData>& imu_data
) {
    EstimationResult result;
    
    // Create new frame first
    m_current_frame = CreateFrame(image, timestamp);
    
    // Process IMU data and compute preintegration
    if (!imu_data.empty() && m_previous_frame) {
        ProcessIMU(imu_data);
    }
    
    if (!m_initialized) {
        // Not initialized yet - accumulate frames for initialization
        if (m_previous_frame) {
            // Track features for initialization
            result.num_tracked = TrackFeatures();
        } else {
            // First frame - detect features
            DetectFeatures();
        }
        
        // Add current frame to window
        m_frame_window.push_back(m_current_frame);
        
        // Maintain window size
        if (static_cast<int>(m_frame_window.size()) > m_window_size) {
            m_frame_window.erase(m_frame_window.begin());
        }
        
        // Try to initialize when window is full
        if (static_cast<int>(m_frame_window.size()) == m_window_size) {
            bool init_result = TryInitialize();
            
            if (init_result) {
                result.init_success = true;
                m_initialized = true;
                
                // Set initialization keyframes
                auto first_frame = m_frame_window.front();
                auto last_frame = m_frame_window.back();
                
                first_frame->SetKeyframe(true);
                last_frame->SetKeyframe(true);
                
                m_keyframes.push_back(first_frame);
                m_keyframes.push_back(last_frame);
                m_last_keyframe = last_frame;
                
                LOG_INFO("Set keyframes: {} and {} (last_keyframe={})", 
                         first_frame->GetFrameId(), last_frame->GetFrameId(), 
                         m_last_keyframe->GetFrameId());
                
                // Process intermediate frames: interpolate poses, link MapPoints, run PnP
                ProcessIntermediateFrames();
            } else {
                if (m_frame_window.size() >= 2) {
                    auto first_frame = m_frame_window.front();
                    auto last_frame = m_frame_window.back();
                    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
                    
                    if (parallax >= m_min_parallax) {
                        result.init_ready = true;
                        result.init_success = true;
                    }
                }
            }
        }
    } else {
        // Already initialized - normal tracking
        int num_tracked = TrackFeatures();
        
        // Link MapPoints from previous frame to current frame
        LinkMapPointsFromPreviousFrame();
        
        // Count valid MapPoints
        int valid_mp_count = 0;
        const auto& features = m_current_frame->GetFeatures();
        for (size_t j = 0; j < features.size(); ++j) {
            auto mp = m_current_frame->GetMapPoint(static_cast<int>(j));
            if (mp && !mp->IsBad()) {
                valid_mp_count++;
            }
        }
        
        // Pose estimation using PnP
        if (valid_mp_count >= 6) {
            // Initialize with previous pose as prior
            m_current_frame->SetTwb(m_previous_frame->GetTwb());
            
            // Run PnP optimization
            Optimizer optimizer;
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            // Compute parallax from last keyframe
            float parallax_from_kf = 0.0f;
            if (m_last_keyframe) {
                parallax_from_kf = ComputeParallax(m_last_keyframe, m_current_frame);
            }
            
            if (pnp_result.success) {
                m_current_pose = m_current_frame->GetTwb();
                LOG_INFO("Frame {}: PnP {} in/{} out, reproj {:.2f}px, parallax {:.1f}px",
                         m_current_frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                         pnp_result.final_cost, parallax_from_kf);
                result.success = true;
            } else {
                LOG_WARN("Frame {}: PnP failed", m_current_frame->GetFrameId());
                result.success = false;
            }
        } else {
            LOG_WARN("Frame {}: Not enough MapPoints ({}) for PnP", 
                     m_current_frame->GetFrameId(), valid_mp_count);
            result.success = false;
        }
        
        if (m_current_frame->GetFeatureCount() < 100) {
            DetectFeatures();
        }
        
        if (ShouldCreateKeyframe()) {
            CreateKeyframe();
        }
        
        result.num_tracked = num_tracked;
    }
    
    // Update state
    result.pose = m_current_pose;
    result.num_features = m_current_frame->GetFeatureCount();
    m_all_frames.push_back(m_current_frame);
    m_previous_frame = m_current_frame;
    
    return result;
}

void Estimator::ProcessIMU(const std::vector<IMUData>& imu_data) {
    if (imu_data.empty() || !m_current_frame || !m_previous_frame) {
        return;
    }
    
    // Get current bias estimates from previous frame
    Eigen::Vector3f accel_bias = m_previous_frame->GetAccelBias();
    Eigen::Vector3f gyro_bias = m_previous_frame->GetGyroBias();
    
    // Set bias in preintegrator
    m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
    
    // Get time range
    double start_time = imu_data.front().timestamp;
    double end_time = imu_data.back().timestamp;
    
    // Compute preintegration from last frame
    auto preint_from_last_frame = m_imu_preintegrator->Preintegrate(
        imu_data, start_time, end_time
    );
    
    // Store preintegration in current frame
    if (preint_from_last_frame) {
        m_current_frame->SetIMUPreintegrationFromLastFrame(preint_from_last_frame);
        
        // Copy bias from previous frame
        m_current_frame->SetAccelBias(accel_bias);
        m_current_frame->SetGyroBias(gyro_bias);
    }
    
    // Accumulate IMU data since last keyframe
    m_imu_since_last_keyframe.insert(
        m_imu_since_last_keyframe.end(),
        imu_data.begin(),
        imu_data.end()
    );
}

bool Estimator::TryInitialize() {
    if (m_frame_window.size() < 2) {
        return false;
    }
    
    // Get first and last frames in window
    auto first_frame = m_frame_window.front();
    auto last_frame = m_frame_window.back();
    
    // Compute parallax between first and last frames
    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
    
    // Check if parallax is sufficient
    if (parallax < m_min_parallax) {
        // Insufficient parallax - silently continue
        return false;
    }
    
    // Step 1: Select features with sufficient observations
    auto selected_features = m_initializer->SelectFeaturesForInit(m_frame_window);
    
    if (selected_features.empty()) {
        return false;
    }
    
    // Step 2: Try monocular initialization
    InitializationResult init_result;
    bool init_success = m_initializer->TryMonocularInitialization(m_frame_window, init_result);
    
    if (!init_success) {
        return false;
    }
    
    // Step 3: Store initialization results
    m_initialized_points = init_result.points3d;
    
    // Create pose matrices (Frame 1 at origin, Frame 2 with [R|t])
    m_init_poses.clear();
    m_init_poses.resize(2);
    
    // T_w1 = Identity (Frame 1 is at world origin)
    m_init_poses[0] = Eigen::Matrix4f::Identity();
    
    // T_w2 = [R | t]
    //        [0 | 1]
    m_init_poses[1] = Eigen::Matrix4f::Identity();
    m_init_poses[1].block<3, 3>(0, 0) = init_result.R;
    m_init_poses[1].block<3, 1>(0, 3) = init_result.t;
    
    // Return true to signal initialization is ready
    return true;
}

void Estimator::Reset() {
    m_current_frame = nullptr;
    m_previous_frame = nullptr;
    m_last_keyframe = nullptr;
    m_all_frames.clear();
    m_keyframes.clear();
    m_frame_window.clear();
    m_frame_id_counter = 0;
    m_initialized = false;
    m_current_pose = Eigen::Matrix4f::Identity();
}

std::shared_ptr<Frame> Estimator::CreateFrame(const cv::Mat& image, double timestamp) {
    const auto& config = ConfigUtils::GetInstance();
    
    // Frame constructor: timestamp (long long in nanoseconds), frame_id, image, width, height
    long long timestamp_ns = static_cast<long long>(timestamp * 1e9);
    auto frame = std::make_shared<Frame>(
        timestamp_ns,
        m_frame_id_counter++,
        image,
        config.camera_width,
        config.camera_height
    );
    
    // Set grid parameters
    frame->SetGridParameters(
        config.grid_cols,
        config.grid_rows,
        config.max_features_per_grid
    );
    
    return frame;
}

int Estimator::TrackFeatures() {
    if (!m_previous_frame || !m_current_frame) {
        return 0;
    }
    
    // Track features using feature tracker
    m_feature_tracker->TrackFeatures(m_current_frame, m_previous_frame);
    
    // Get tracking stats
    int num_tracked, num_detected;
    m_feature_tracker->GetTrackingStats(num_tracked, num_detected);
    
    return num_tracked;
}

int Estimator::DetectFeatures() {
    if (!m_current_frame) {
        return 0;
    }
    
    // Detect features (pass nullptr as previous frame)
    m_feature_tracker->TrackFeatures(m_current_frame, nullptr);
    
    return m_current_frame->GetFeatureCount();
}

bool Estimator::ShouldCreateKeyframe() {
    if (!m_current_frame || !m_last_keyframe) {
        return false;
    }
    
    // Get parallax threshold from config
    const auto& config = ConfigUtils::GetInstance();
    float parallax_threshold = config.tracking_min_parallax_for_keyframe;
    
    // Compute parallax between last keyframe and current frame
    float parallax = ComputeParallax(m_last_keyframe, m_current_frame);
    
    // Create keyframe if parallax exceeds threshold
    if (parallax >= parallax_threshold) {
        LOG_INFO("Parallax {:.2f} >= {:.2f}, creating new keyframe", parallax, parallax_threshold);
        return true;
    }
    
    return false;
}

void Estimator::CreateKeyframe() {
    if (!m_current_frame) {
        return;
    }
    
    // Get previous keyframe before updating m_last_keyframe
    auto prev_keyframe = m_last_keyframe;
    
    // Mark current frame as keyframe
    m_current_frame->SetKeyframe(true);
    m_keyframes.push_back(m_current_frame);
    m_last_keyframe = m_current_frame;
    
    // Maintain keyframe window size (remove oldest keyframes from front)
    const int max_keyframes = 10;  // TODO: make configurable
    while (static_cast<int>(m_keyframes.size()) > max_keyframes) {
        m_keyframes.erase(m_keyframes.begin());  // Remove oldest (front)
    }
    
    // Triangulate new MapPoints between previous keyframe and current keyframe
    int new_points = 0;
    if (prev_keyframe) {
        new_points = TriangulateNewMapPoints(prev_keyframe, m_current_frame);
    }
    
    // Run local BA if we have new MapPoints
    if (new_points > 0 && m_keyframes.size() >= 2) {
        LOG_INFO("Running local BA with {} keyframes...", m_keyframes.size());
        Optimizer optimizer;
        BAResult ba_result = optimizer.RunBA(m_keyframes, true, false);  // Fix first pose only
        LOG_INFO("Local BA done: {} inliers, {} outliers", 
                 ba_result.num_inliers, ba_result.num_outliers);
    }
    
    LOG_INFO("Created keyframe {} (total: {}), triangulated {} new MapPoints", 
             m_current_frame->GetFrameId(), m_keyframes.size(), new_points);
}

void Estimator::LinkMapPointsFromPreviousFrame() {
    if (!m_current_frame || !m_previous_frame) {
        return;
    }
    
    const auto& curr_features = m_current_frame->GetFeatures();
    const auto& prev_features = m_previous_frame->GetFeatures();
    
    // Build map from feature_id to index in previous frame
    // Only include valid features (not outliers from PnP)
    std::unordered_map<int, size_t> prev_feature_map;
    int prev_with_mp = 0;
    for (size_t i = 0; i < prev_features.size(); ++i) {
        if (!prev_features[i]->IsValid()) continue;  // Skip outliers
        prev_feature_map[prev_features[i]->GetFeatureId()] = i;
        auto mp = m_previous_frame->GetMapPoint(static_cast<int>(i));
        if (mp && !mp->IsBad()) prev_with_mp++;
    }
    
    // Link MapPoints based on matching feature IDs
    int linked_count = 0;
    for (size_t i = 0; i < curr_features.size(); ++i) {
        int feat_id = curr_features[i]->GetFeatureId();
        
        auto it = prev_feature_map.find(feat_id);
        if (it != prev_feature_map.end()) {
            size_t prev_idx = it->second;
            auto mp = m_previous_frame->GetMapPoint(static_cast<int>(prev_idx));
            
            if (mp && !mp->IsBad()) {
                m_current_frame->SetMapPoint(static_cast<int>(i), mp);
                linked_count++;
            }
        }
    }
    
    LOG_INFO("LinkMapPoints: prev had {} MPs, linked {} to current frame", prev_with_mp, linked_count);
}

void Estimator::ProcessIntermediateFrames() {
    if (m_frame_window.size() < 3) {
        return;  // No intermediate frames
    }
    
    auto first_kf = m_frame_window.front();
    auto last_kf = m_frame_window.back();
    
    Eigen::Matrix4f T_first = first_kf->GetTwb();
    Eigen::Matrix4f T_last = last_kf->GetTwb();
    
    // Extract rotation and translation
    Eigen::Matrix3f R_first = T_first.block<3, 3>(0, 0);
    Eigen::Vector3f t_first = T_first.block<3, 1>(0, 3);
    Eigen::Matrix3f R_last = T_last.block<3, 3>(0, 0);
    Eigen::Vector3f t_last = T_last.block<3, 1>(0, 3);
    
    // Compute relative rotation using Rodrigues
    Eigen::Matrix3f R_rel = R_first.transpose() * R_last;
    Eigen::AngleAxisf aa(R_rel);
    Eigen::Vector3f axis = aa.axis();
    float angle = aa.angle();
    
    int n_frames = static_cast<int>(m_frame_window.size());
    
    LOG_INFO("Processing {} intermediate frames...", n_frames - 2);
    
    // Process intermediate frames (skip first and last which are keyframes)
    for (int i = 1; i < n_frames - 1; ++i) {
        auto& frame = m_frame_window[i];
        auto& prev_frame = m_frame_window[i - 1];
        
        // Interpolate pose (linear interpolation for translation, slerp for rotation)
        float alpha = static_cast<float>(i) / static_cast<float>(n_frames - 1);
        
        // Interpolate rotation using axis-angle
        float interp_angle = alpha * angle;
        Eigen::Matrix3f R_interp = R_first * Eigen::AngleAxisf(interp_angle, axis).toRotationMatrix();
        
        // Interpolate translation
        Eigen::Vector3f t_interp = (1.0f - alpha) * t_first + alpha * t_last;
        
        // Set interpolated pose
        Eigen::Matrix4f T_interp = Eigen::Matrix4f::Identity();
        T_interp.block<3, 3>(0, 0) = R_interp;
        T_interp.block<3, 1>(0, 3) = t_interp;
        frame->SetTwb(T_interp);
        
        // Link MapPoints from previous frame
        const auto& curr_features = frame->GetFeatures();
        const auto& prev_features = prev_frame->GetFeatures();
        
        std::unordered_map<int, size_t> prev_feature_map;
        for (size_t j = 0; j < prev_features.size(); ++j) {
            prev_feature_map[prev_features[j]->GetFeatureId()] = j;
        }
        
        int linked = 0;
        for (size_t j = 0; j < curr_features.size(); ++j) {
            int feat_id = curr_features[j]->GetFeatureId();
            auto it = prev_feature_map.find(feat_id);
            if (it != prev_feature_map.end()) {
                auto mp = prev_frame->GetMapPoint(static_cast<int>(it->second));
                if (mp && !mp->IsBad()) {
                    frame->SetMapPoint(static_cast<int>(j), mp);
                    linked++;
                }
            }
        }
        
        // Run PnP to refine pose
        if (linked >= 6) {
            Optimizer optimizer;
            PnPResult pnp_result = optimizer.SolvePnP(frame);
            LOG_INFO("  Frame {}: interpolated -> PnP {} in/{} out, reproj {:.2f}px",
                     frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                     pnp_result.final_cost);
        }
    }
    
    // Run BA on all frames in window
    LOG_INFO("Running BA on {} frames in initialization window...", n_frames);
    Optimizer optimizer;
    BAResult ba_result = optimizer.RunBA(m_frame_window, true, false);
    LOG_INFO("Init window BA done: {} inliers, {} outliers", 
             ba_result.num_inliers, ba_result.num_outliers);
    
    // Mark all intermediate frames as keyframes and add to keyframe list
    for (int i = 1; i < n_frames - 1; ++i) {
        auto& frame = m_frame_window[i];
        frame->SetKeyframe(true);
        // Insert in order (after first keyframe, before last keyframe)
        m_keyframes.insert(m_keyframes.end() - 1, frame);
    }
    LOG_INFO("Added {} intermediate frames as keyframes (total: {})", 
             n_frames - 2, m_keyframes.size());
}

float Estimator::ComputeParallax(
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
    
    // Build map from feature_id to pixel coord in frame1
    std::unordered_map<int, cv::Point2f> feat1_map;
    for (const auto& feat : features1) {
        feat1_map[feat->GetFeatureId()] = feat->GetPixelCoord();
    }
    
    // Find correspondences and compute parallax
    std::vector<float> parallaxes;
    parallaxes.reserve(features2.size());
    
    for (const auto& feat2 : features2) {
        auto it = feat1_map.find(feat2->GetFeatureId());
        if (it != feat1_map.end()) {
            const cv::Point2f& pt1 = it->second;
            cv::Point2f pt2 = feat2->GetPixelCoord();
            
            float dx = pt2.x - pt1.x;
            float dy = pt2.y - pt1.y;
            float parallax = std::sqrt(dx * dx + dy * dy);
            
            parallaxes.push_back(parallax);
        }
    }
    
    if (parallaxes.empty()) {
        return 0.0f;
    }
    
    // Return median parallax (more robust than mean)
    std::sort(parallaxes.begin(), parallaxes.end());
    size_t mid = parallaxes.size() / 2;
    
    if (parallaxes.size() % 2 == 0) {
        return (parallaxes[mid - 1] + parallaxes[mid]) / 2.0f;
    } else {
        return parallaxes[mid];
    }
}

bool Estimator::TriangulateSinglePoint(
    const Eigen::Vector3f& bearing1,
    const Eigen::Vector3f& bearing2,
    const Eigen::Matrix4f& T1w,
    const Eigen::Matrix4f& T2w,
    Eigen::Vector3f& point3d
) const {
    // Get camera centers in world frame
    // T_cw means world to camera, so camera center = -R^T * t
    Eigen::Matrix3f R1 = T1w.block<3, 3>(0, 0);
    Eigen::Vector3f t1 = T1w.block<3, 1>(0, 3);
    Eigen::Vector3f C1 = -R1.transpose() * t1;  // Camera 1 center in world
    
    Eigen::Matrix3f R2 = T2w.block<3, 3>(0, 0);
    Eigen::Vector3f t2 = T2w.block<3, 1>(0, 3);
    Eigen::Vector3f C2 = -R2.transpose() * t2;  // Camera 2 center in world
    
    // Transform bearing vectors to world frame
    Eigen::Vector3f d1 = R1.transpose() * bearing1;  // Ray direction in world
    Eigen::Vector3f d2 = R2.transpose() * bearing2;
    
    d1.normalize();
    d2.normalize();
    
    // Solve for closest point between two rays using mid-point method
    // Ray1: P = C1 + λ1 * d1
    // Ray2: P = C2 + λ2 * d2
    
    Eigen::Vector3f w0 = C1 - C2;
    float a = d1.dot(d1);
    float b = d1.dot(d2);
    float c = d2.dot(d2);
    float d = d1.dot(w0);
    float e = d2.dot(w0);
    
    float denom = a * c - b * b;
    if (std::abs(denom) < 1e-8f) {
        LOG_WARN("  Tri fail: rays parallel, denom={:.2e}", denom);
        return false;  // Rays are parallel
    }
    
    float lambda1 = (b * e - c * d) / denom;
    float lambda2 = (a * e - b * d) / denom;
    
    // Check positive depth
    if (lambda1 < 0.01f || lambda2 < 0.01f) {
        LOG_WARN("  Tri fail: negative depth, lambda1={:.4f}, lambda2={:.4f}", lambda1, lambda2);
        return false;
    }
    
    // Compute points on each ray
    Eigen::Vector3f P1 = C1 + lambda1 * d1;
    Eigen::Vector3f P2 = C2 + lambda2 * d2;
    
    // Mid-point
    point3d = (P1 + P2) / 2.0f;
    
    // Check reprojection (optional, for quality)
    // For small baseline, allow larger ratio since measurement noise dominates
    float dist = (P1 - P2).norm();
    float baseline = (C1 - C2).norm();
    
    // Use absolute distance threshold instead of ratio for small baselines
    // Also check that the point is not too far (depth sanity check)
    float max_depth = 50.0f;  // Maximum depth in meters
    float depth1 = lambda1;
    float depth2 = lambda2;
    
    if (depth1 > max_depth || depth2 > max_depth) {
        LOG_WARN("  Tri fail: depth too large, d1={:.2f}, d2={:.2f}", depth1, depth2);
        return false;
    }
    
    // For intersection quality, use absolute threshold or relaxed ratio
    float abs_threshold = 0.1f;  // 10cm absolute threshold
    float ratio_threshold = 1.0f;  // 100% of baseline
    
    if (dist > abs_threshold && dist > ratio_threshold * baseline) {
        LOG_WARN("  Tri fail: bad intersection, dist={:.4f}, baseline={:.4f}, ratio={:.2f}", 
                 dist, baseline, dist/baseline);
        return false;  // Rays don't intersect well
    }
    
    return true;
}

int Estimator::TriangulateNewMapPoints(
    const std::shared_ptr<Frame>& kf1,
    const std::shared_ptr<Frame>& kf2
) {
    if (!kf1 || !kf2) {
        return 0;
    }
    
    const auto& features1 = kf1->GetFeatures();
    const auto& features2 = kf2->GetFeatures();
    
    LOG_INFO("Triangulation: kf1={} ({} feats), kf2={} ({} feats)",
             kf1->GetFrameId(), features1.size(), kf2->GetFrameId(), features2.size());
    
    // Build map: feature_id -> (index in kf1, feature pointer)
    // Only include valid features (not outliers)
    std::unordered_map<int, std::pair<int, std::shared_ptr<Feature>>> kf1_map;
    for (size_t i = 0; i < features1.size(); ++i) {
        if (!features1[i]->IsValid()) continue;  // Skip outliers
        int feat_id = features1[i]->GetFeatureId();
        kf1_map[feat_id] = {static_cast<int>(i), features1[i]};
    }
    
    // Get poses (world to camera)
    // GetTwc() returns camera to world, so we need to invert it
    Eigen::Matrix4f T1w = kf1->GetTwc().inverse();  // World to camera1
    Eigen::Matrix4f T2w = kf2->GetTwc().inverse();  // World to camera2
    
    // Debug: print baseline
    Eigen::Matrix3f R1 = T1w.block<3, 3>(0, 0);
    Eigen::Vector3f t1 = T1w.block<3, 1>(0, 3);
    Eigen::Vector3f C1 = -R1.transpose() * t1;
    
    Eigen::Matrix3f R2 = T2w.block<3, 3>(0, 0);
    Eigen::Vector3f t2 = T2w.block<3, 1>(0, 3);
    Eigen::Vector3f C2 = -R2.transpose() * t2;
    
    float baseline = (C1 - C2).norm();
    LOG_INFO("  Baseline: {:.6f}, C1=({:.4f},{:.4f},{:.4f}), C2=({:.4f},{:.4f},{:.4f})",
             baseline, C1.x(), C1.y(), C1.z(), C2.x(), C2.y(), C2.z());
    
    int triangulated_count = 0;
    int matched_count = 0;
    int already_has_mp = 0;
    int triangulation_failed = 0;
    
    for (size_t i2 = 0; i2 < features2.size(); ++i2) {
        // Skip invalid features (outliers)
        if (!features2[i2]->IsValid()) continue;
        
        int feat_id = features2[i2]->GetFeatureId();
        
        // Check if this feature exists in kf1
        auto it = kf1_map.find(feat_id);
        if (it == kf1_map.end()) {
            continue;
        }
        
        matched_count++;
        int i1 = it->second.first;
        
        // Check if already has MapPoint
        auto existing_mp = kf2->GetMapPoint(static_cast<int>(i2));
        if (existing_mp && !existing_mp->IsBad()) {
            already_has_mp++;
            continue;  // Already has valid MapPoint
        }
        
        // Get bearing vectors
        Eigen::Vector3f bearing1 = features1[i1]->GetBearing();
        Eigen::Vector3f bearing2 = features2[i2]->GetBearing();
        
        // Triangulate
        Eigen::Vector3f point3d;
        if (!TriangulateSinglePoint(bearing1, bearing2, T1w, T2w, point3d)) {
            triangulation_failed++;
            continue;
        }
        
        // Create new MapPoint
        auto mp = std::make_shared<MapPoint>(point3d);
        mp->SetTriangulated(true);
        
        // Add observations to MapPoint
        mp->AddObservation(kf1, i1);
        mp->AddObservation(kf2, static_cast<int>(i2));
        
        // Register MapPoint to keyframes
        kf1->SetMapPoint(i1, mp);
        kf2->SetMapPoint(static_cast<int>(i2), mp);
        
        triangulated_count++;
    }
    
    LOG_INFO("  Matched: {}, already_has_mp: {}, tri_failed: {}, success: {}",
             matched_count, already_has_mp, triangulation_failed, triangulated_count);
    
    return triangulated_count;
}

} // namespace vio_360
