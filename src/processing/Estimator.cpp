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
    , m_imu_initialized(false)
    , m_current_pose(Eigen::Matrix4f::Identity())
    , m_transform_from_last(Eigen::Matrix4f::Identity())
    , m_gravity(0, 0, -9.81f)
    , m_gyro_bias(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_scale(1.0) {
    
    // Initialize camera
    const auto& config = ConfigUtils::GetInstance();
    m_camera = std::make_shared<Camera>(
        config.camera_width,
        config.camera_height
    );
    m_boundary_margin = config.camera_boundary_margin;
    
    // Initialize feature tracker
    m_feature_tracker = std::make_unique<FeatureTracker>(
        m_camera,
        config.max_features,
        config.min_distance,
        config.quality_level,
        config.camera_boundary_margin
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
        
        LOG_DEBUG("  Window size: {}/{}, trying init...", m_frame_window.size(), m_window_size);
        
        // Try to initialize when window is full
        if (static_cast<int>(m_frame_window.size()) == m_window_size) {
            bool init_result = TryInitialize();
            
            if (init_result) {
                // Initialization succeeded!
                result.init_success = true;
                m_initialized = true;
                
                LOG_INFO("Initialization SUCCESS!");
                
                // Set initialization keyframes
                auto first_frame = m_frame_window.front();
                auto last_frame = m_frame_window.back();
                
                first_frame->SetKeyframe(true);
                last_frame->SetKeyframe(true);
                
                // Compute preintegration from first_frame to last_frame for IMU init
                if (!m_imu_since_last_keyframe.empty()) {
                    Eigen::Vector3f gyro_bias = first_frame->GetGyroBias();
                    Eigen::Vector3f accel_bias = first_frame->GetAccelBias();
                    m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
                    
                    double start_time = m_imu_since_last_keyframe.front().timestamp;
                    double end_time = m_imu_since_last_keyframe.back().timestamp;
                    
                    auto preint_from_first_kf = m_imu_preintegrator->Preintegrate(
                        m_imu_since_last_keyframe, start_time, end_time
                    );
                    
                    if (preint_from_first_kf) {
                        last_frame->SetIMUPreintegrationFromLastKeyframe(preint_from_first_kf);
                        LOG_INFO("  IMU preintegration: kf {} -> kf {}: dt={:.3f}s, {} samples",
                                 first_frame->GetFrameId(), last_frame->GetFrameId(),
                                 preint_from_first_kf->dt_total, m_imu_since_last_keyframe.size());
                    }
                    m_imu_since_last_keyframe.clear();
                }
                
                m_keyframes.push_back(first_frame);
                m_keyframes.push_back(last_frame);
                m_last_keyframe = last_frame;
            } else {
                // Check if ready for initialization (sufficient parallax)
                if (m_frame_window.size() >= 2) {
                    auto first_frame = m_frame_window.front();
                    auto last_frame = m_frame_window.back();
                    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
                    
                    if (parallax >= m_min_parallax) {
                        result.init_ready = true;
                        // Note: Don't set init_success here - only on actual success
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
        
        // Initialize pose: use IMU prediction if available, otherwise use previous pose
        bool imu_predicted = false;
        if (m_imu_initialized) {
            imu_predicted = PredictStateWithIMU();
        }
        if (!imu_predicted) {
            m_current_frame->SetTwb(m_previous_frame->GetTwb());
        }
        
        // Pose estimation using PnP
        if (valid_mp_count >= 6) {
            // Run PnP optimization
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            // Compute parallax from last keyframe
            float parallax_from_kf = 0.0f;
            if (m_last_keyframe) {
                parallax_from_kf = ComputeParallax(m_last_keyframe, m_current_frame);
            }
            
            if (pnp_result.success) {
                // Set reference keyframe FIRST (before GetTwb)
                if (m_last_keyframe && !m_current_frame->IsKeyframe()) {
                    m_current_frame->SetReferenceKeyframe(m_last_keyframe);
                }
                
                m_current_pose = m_current_frame->GetTwb();
                LOG_INFO("Frame {}: PnP {} in/{} out, cost {:.2f}->{:.2f}",
                         m_current_frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                         pnp_result.initial_cost, pnp_result.final_cost);
                
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
                
                // Process all frames in window: interpolate poses, run PnP, BA, compute preintegration
                // This creates keyframes for ALL frames in the window (not just first and last)
                ProcessIntermediateFrames();
                
                // Now run IMU initialization with all 10 keyframes
                if (!m_imu_initialized && m_keyframes.size() >= 3) {
                    TryInitializeIMU();
                }
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
        
        // Initialize pose: use IMU prediction if available, otherwise constant velocity model
        bool imu_predicted = false;
        if (m_imu_initialized) {
            imu_predicted = PredictStateWithIMU();
        }
        
        if (!imu_predicted) {
            // Fallback to constant velocity motion model
            if (m_transform_from_last.isIdentity(1e-6)) {
                // First frame after initialization, use previous pose
                m_current_frame->SetTwb(m_previous_frame->GetTwb());
            } else {
                // Apply motion model: predict current pose from velocity
                Eigen::Matrix4f predicted_pose = m_previous_frame->GetTwb() * m_transform_from_last;
                m_current_frame->SetTwb(predicted_pose);
            }
        }
        
        // Pose estimation using PnP
        if (valid_mp_count >= 6) {
            // Run PnP optimization
            const auto& config = ConfigUtils::GetInstance();
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            // Compute parallax from last keyframe
            float parallax_from_kf = 0.0f;
            if (m_last_keyframe) {
                parallax_from_kf = ComputeParallax(m_last_keyframe, m_current_frame);
            }
            
            if (pnp_result.success) {
                // Set reference keyframe FIRST (before GetTwb)
                if (m_last_keyframe && !m_current_frame->IsKeyframe()) {
                    m_current_frame->SetReferenceKeyframe(m_last_keyframe);
                }
                
                // Update current pose and transform for next frame
                m_current_pose = m_current_frame->GetTwb();
                m_transform_from_last = m_previous_frame->GetTwb().inverse() * m_current_pose;
                
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

bool Estimator::PredictStateWithIMU() {
    if (!m_imu_initialized || !m_current_frame || !m_previous_frame) {
        return false;
    }
    
    // Get IMU preintegration from last frame
    auto preint = m_current_frame->GetIMUPreintegrationFromLastFrame();
    if (!preint || !preint->IsValid()) {
        return false;
    }
    
    // Get previous frame state
    Eigen::Matrix4f T_wb_prev = m_previous_frame->GetTwb();
    Eigen::Matrix3f R_wb_prev = T_wb_prev.block<3, 3>(0, 0);
    Eigen::Vector3f t_wb_prev = T_wb_prev.block<3, 1>(0, 3);
    Eigen::Vector3f v_prev = m_previous_frame->GetVelocity();
    
    // Get preintegration values
    float dt = static_cast<float>(preint->dt_total);
    Eigen::Matrix3f delta_R = preint->delta_R;
    Eigen::Vector3f delta_V = preint->delta_V;
    Eigen::Vector3f delta_P = preint->delta_P;
    
    // Predict current state using IMU preintegration
    // R_new = R_prev * delta_R
    Eigen::Matrix3f R_wb_curr = R_wb_prev * delta_R;
    
    // V_new = V_prev + g * dt + R_prev * delta_V
    Eigen::Vector3f v_curr = v_prev + m_gravity * dt + R_wb_prev * delta_V;
    
    // P_new = P_prev + V_prev * dt + 0.5 * g * dt^2 + R_prev * delta_P
    Eigen::Vector3f t_wb_curr = t_wb_prev + v_prev * dt + 0.5f * m_gravity * dt * dt + R_wb_prev * delta_P;
    
    // Set predicted state
    Eigen::Matrix4f T_wb_curr = Eigen::Matrix4f::Identity();
    T_wb_curr.block<3, 3>(0, 0) = R_wb_curr;
    T_wb_curr.block<3, 1>(0, 3) = t_wb_curr;
    
    m_current_frame->SetTwb(T_wb_curr);
    m_current_frame->SetVelocity(v_curr);
    
    return true;
}

bool Estimator::TryInitialize() {
    if (m_frame_window.size() < 2) {
        LOG_DEBUG("[Init] Window size {} < 2, skip", m_frame_window.size());
        return false;
    }
    
    // Get first and last frames in window
    auto first_frame = m_frame_window.front();
    auto last_frame = m_frame_window.back();
    
    // Compute parallax between first and last frames
    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
    
    LOG_DEBUG("[Init] Frame {}->{}: parallax={:.2f}px (min={:.2f})", 
             first_frame->GetFrameId(), last_frame->GetFrameId(), 
             parallax, m_min_parallax);
    
    // Check if parallax is sufficient
    if (parallax < m_min_parallax) {
        LOG_DEBUG("[Init] Insufficient parallax, waiting...");
        return false;
    }
    
    // Step 1: Select features with sufficient observations
    auto selected_features = m_initializer->SelectFeaturesForInit(m_frame_window);
    
    LOG_DEBUG("[Init] Selected {} features for initialization", selected_features.size());
    
    if (selected_features.empty()) {
        LOG_WARN("[Init] No features selected, skip");
        return false;
    }
    
    // Step 2: Try monocular initialization
    LOG_DEBUG("[Init] Attempting monocular initialization...");
    InitializationResult init_result;
    bool init_success = m_initializer->TryMonocularInitialization(m_frame_window, init_result);
    
    if (!init_success) {
        LOG_WARN("[Init] Monocular initialization failed");
        return false;
    }
    
    LOG_INFO("Monocular init: {} MapPoints created", init_result.initialized_mappoints.size());
    
    // Mark initialization MapPoints as marginalized (they define the scale)
    for (const auto& mp : init_result.initialized_mappoints) {
        if (mp) {
            mp->SetMarginalized(true);
        }
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
    m_transform_from_last = Eigen::Matrix4f::Identity();
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
    
    // Set camera-to-body extrinsic transformation
    frame->SetTBC(config.T_BC);
    
    // Copy bias from previous frame (after IMU initialization)
    if (m_imu_initialized && m_previous_frame) {
        frame->SetGyroBias(m_previous_frame->GetGyroBias());
        frame->SetAccelBias(m_previous_frame->GetAccelBias());
    }
    
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
        LOG_DEBUG("Parallax {:.2f} >= {:.2f}, creating new keyframe", parallax, parallax_threshold);
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
    
    // Compute and store preintegration from last keyframe BEFORE updating m_last_keyframe
    if (prev_keyframe && !m_imu_since_last_keyframe.empty()) {
        // Get bias from previous keyframe
        Eigen::Vector3f gyro_bias = prev_keyframe->GetGyroBias();
        Eigen::Vector3f accel_bias = prev_keyframe->GetAccelBias();
        m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
        
        // Compute preintegration from accumulated IMU data
        double start_time = m_imu_since_last_keyframe.front().timestamp;
        double end_time = m_imu_since_last_keyframe.back().timestamp;
        
        auto preint_from_last_kf = m_imu_preintegrator->Preintegrate(
            m_imu_since_last_keyframe, start_time, end_time
        );
        
        if (preint_from_last_kf) {
            m_current_frame->SetIMUPreintegrationFromLastKeyframe(preint_from_last_kf);
            LOG_DEBUG("  IMU: preintegration from kf {} to kf {}: dt={:.3f}s, {} measurements",
                     prev_keyframe->GetFrameId(), m_current_frame->GetFrameId(),
                     preint_from_last_kf->dt_total, m_imu_since_last_keyframe.size());
        }
    }
    
    // Clear accumulated IMU data for next keyframe
    m_imu_since_last_keyframe.clear();
    
    // Mark current frame as keyframe
    m_current_frame->SetKeyframe(true);
    m_keyframes.push_back(m_current_frame);
    m_all_keyframes.push_back(m_current_frame);  // Also add to all keyframes list
    m_last_keyframe = m_current_frame;
    
    // Add observations to MapPoints for this keyframe
    // (MapPoints linked from previous frames now get this keyframe as observer)
    const auto& features = m_current_frame->GetFeatures();
    int obs_added = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        auto mp = m_current_frame->GetMapPoint(static_cast<int>(i));
        if (mp && !mp->IsBad() && features[i] && features[i]->IsValid()) {
            // Check if this keyframe is already observing this MapPoint
            if (!mp->IsObservedByFrame(m_current_frame)) {
                mp->AddObservation(m_current_frame, static_cast<int>(i));
                obs_added++;
            }
        }
    }
    
    // Maintain keyframe window size (remove oldest keyframes from front)
    const int max_keyframes = 10;  // TODO: make configurable
    while (static_cast<int>(m_keyframes.size()) > max_keyframes) {
        auto oldest_keyframe = m_keyframes.front();
        
        // For MapPoints whose reference keyframe is being removed:
        // Transfer reference to another keyframe in the window (keep MapPoint alive)
        const auto& old_kf_features = oldest_keyframe->GetFeatures();
        int transferred_count = 0;
        int deleted_count = 0;
        
        for (size_t i = 0; i < old_kf_features.size(); ++i) {
            auto mp = oldest_keyframe->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                // Check if this keyframe is the reference (origin) for this MapPoint
                if (mp->IsReferenceKeyframe(oldest_keyframe)) {
                    // Find the oldest keyframe in window that observes this MapPoint
                    std::shared_ptr<Frame> new_ref_keyframe = nullptr;
                    
                    // Iterate through keyframes in window (oldest to newest, skipping the one being removed)
                    for (size_t kf_idx = 1; kf_idx < m_keyframes.size(); ++kf_idx) {
                        auto& kf = m_keyframes[kf_idx];
                        if (mp->IsObservedByFrame(kf)) {
                            new_ref_keyframe = kf;
                            break;  // Take the oldest one in window
                        }
                    }
                    
                    if (new_ref_keyframe) {
                        // Transfer reference to new keyframe
                        // Marginalize the MapPoint to preserve scale (its original reference defined the scale)
                        mp->SetReferenceKeyframe(new_ref_keyframe);
                        mp->SetMarginalized(true);  // Marginalize MapPoint when reference keyframe is removed
                        transferred_count++;
                    } else {
                        // No keyframe in window observes this MapPoint, mark as bad
                        mp->SetBad(true);
                        deleted_count++;
                    }
                }
            }
        }
        
        if (transferred_count > 0 || deleted_count > 0) {
            LOG_DEBUG("  Reference transfer: {} MapPoints moved, {} deleted (from removed kf {})", 
                     transferred_count, deleted_count, oldest_keyframe->GetFrameId());
        }
        
        // Clean up observations from MapPoints before removing the keyframe
        const auto& features = oldest_keyframe->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            auto mp = oldest_keyframe->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mp->RemoveObservation(oldest_keyframe);
                // Mark MapPoint as bad if no observations left
                if (mp->GetObservationCount() == 0) {
                    mp->SetBad(true);
                }
            }
        }
        
        m_keyframes.erase(m_keyframes.begin());  // Remove oldest (front)
    }
    
    // Triangulate new MapPoints between previous keyframe and current keyframe
    int new_points = 0;
    if (prev_keyframe) {
        new_points = TriangulateNewMapPoints(prev_keyframe, m_current_frame);
    }
    
    // Run BA if we have new MapPoints
    if (new_points > 0 && m_keyframes.size() >= 2) {
        Optimizer optimizer;
        optimizer.SetCamera(m_camera, m_boundary_margin);
        
        BAResult ba_result;
        if (m_imu_initialized) {
            // Visual-only BA for now (VIBA disabled for debugging)
            ba_result = optimizer.RunLocalBA(m_keyframes);
            
            // // Visual-Inertial BA after IMU initialization
            // ba_result = optimizer.RunVIBA(m_keyframes, m_gravity, true);
            
            // // Get updated bias from last keyframe
            // auto last_kf = m_keyframes.back();
            // Eigen::Vector3f new_gyro_bias = last_kf->GetGyroBias();
            // Eigen::Vector3f new_accel_bias = last_kf->GetAccelBias();
            
            // LOG_INFO("VIBA KF{}: bg=[{:.6f},{:.6f},{:.6f}] ba=[{:.6f},{:.6f},{:.6f}]",
            //          last_kf->GetFrameId(),
            //          new_gyro_bias.x(), new_gyro_bias.y(), new_gyro_bias.z(),
            //          new_accel_bias.x(), new_accel_bias.y(), new_accel_bias.z());
            
            // // Update all preintegrations with new bias
            // UpdatePreintegrationsWithNewBias(new_gyro_bias, new_accel_bias);
        } else {
            // Visual-only Local BA before IMU initialization
            ba_result = optimizer.RunLocalBA(m_keyframes);
        }
        
        // Update m_current_pose with BA-optimized pose
        m_current_pose = m_current_frame->GetTwb();
        
        // Reset m_transform_from_last to identity after BA
        // Next frame will start from the BA-optimized keyframe pose
        m_transform_from_last = Eigen::Matrix4f::Identity();
    }
    
    // Try IMU initialization after VO is initialized and we have enough keyframes
    if (m_initialized && !m_imu_initialized && m_keyframes.size() >= 3) {
        TryInitializeIMU();
    }
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
    
    LOG_DEBUG("LinkMapPoints: prev had {} MPs, linked {} to current frame", prev_with_mp, linked_count);
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
    
    LOG_DEBUG("Processing {} intermediate frames to create keyframes...", n_frames - 2);
    
    // Build global feature ID to MapPoint map from first_kf and last_kf (the two frames with triangulated MapPoints)
    std::unordered_map<int, std::shared_ptr<MapPoint>> feature_to_mappoint;
    
    // Collect from first keyframe
    const auto& first_features = first_kf->GetFeatures();
    for (size_t j = 0; j < first_features.size(); ++j) {
        auto mp = first_kf->GetMapPoint(static_cast<int>(j));
        if (mp && !mp->IsBad()) {
            feature_to_mappoint[first_features[j]->GetFeatureId()] = mp;
        }
    }
    
    // Collect from last keyframe (may add new MapPoints or override)
    const auto& last_features = last_kf->GetFeatures();
    for (size_t j = 0; j < last_features.size(); ++j) {
        auto mp = last_kf->GetMapPoint(static_cast<int>(j));
        if (mp && !mp->IsBad()) {
            int feat_id = last_features[j]->GetFeatureId();
            if (feature_to_mappoint.find(feat_id) == feature_to_mappoint.end()) {
                feature_to_mappoint[feat_id] = mp;
            }
        }
    }
    
    LOG_DEBUG("Found {} unique MapPoints from first/last keyframes for observation propagation", 
             feature_to_mappoint.size());
    
    // Process intermediate frames (skip first and last which already have poses)
    for (int i = 1; i < n_frames - 1; ++i) {
        auto& frame = m_frame_window[i];
        
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
        
        // Link MapPoints from global feature->mappoint map (matches feature IDs across all triangulated points)
        const auto& curr_features = frame->GetFeatures();
        
        int linked = 0;
        for (size_t j = 0; j < curr_features.size(); ++j) {
            int feat_id = curr_features[j]->GetFeatureId();
            auto it = feature_to_mappoint.find(feat_id);
            if (it != feature_to_mappoint.end()) {
                auto mp = it->second;
                if (mp && !mp->IsBad()) {
                    frame->SetMapPoint(static_cast<int>(j), mp);
                    mp->AddObservation(frame, static_cast<int>(j));
                    linked++;
                }
            }
        }
        
        LOG_INFO("  Frame {} (idx {}): linked {} MapPoints", frame->GetFrameId(), i, linked);
        
        // Run PnP to refine pose
        if (linked >= 6) {
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(frame);
            LOG_INFO("  Frame {} PnP: {} inliers, {} outliers, reproj={:.2f}px",
                     frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                     pnp_result.final_cost);
        } else {
            LOG_WARN("  Frame {} (idx {}): only {} linked points, skipping PnP", 
                    frame->GetFrameId(), i, linked);
        }
    }
    
    // Run BA on all frames in window (first frame fixed, last frame NOT fixed)
    LOG_INFO("Running BA on {} frames with all MapPoints...", n_frames);
    Optimizer optimizer;
    optimizer.SetCamera(m_camera, m_boundary_margin);
    BAResult ba_result = optimizer.RunBA(m_frame_window, true, false);
    LOG_INFO("Init window BA: {} inliers, {} outliers, cost {:.2f} -> {:.2f}",
             ba_result.num_inliers, ba_result.num_outliers,
             ba_result.initial_cost, ba_result.final_cost);
    
    // Mark all frames as keyframes
    for (int i = 0; i < n_frames; ++i) {
        auto& frame = m_frame_window[i];
        frame->SetKeyframe(true);
    }
    
    // Use preintegration from last frame (already computed per-frame during tracking)
    // In init phase, each frame has preintegration from previous frame
    // For keyframes: just copy frame-to-frame preintegration as keyframe-to-keyframe
    LOG_INFO("Setting up IMU preintegration for {} keyframe pairs...", n_frames - 1);
    int valid_preint_count = 0;
    for (int i = 1; i < n_frames; ++i) {
        auto& curr_kf = m_frame_window[i];
        
        // Get preintegration from last frame (which is also the previous keyframe in init phase)
        auto preint = curr_kf->GetIMUPreintegrationFromLastFrame();
        if (preint) {
            curr_kf->SetIMUPreintegrationFromLastKeyframe(preint);
            valid_preint_count++;
            LOG_INFO("  IMU preint[{}]: kf{} -> kf{}: dt={:.3f}s, dV=({:.3f},{:.3f},{:.3f})",
                     i-1, m_frame_window[i-1]->GetFrameId(), curr_kf->GetFrameId(),
                     preint->dt_total,
                     preint->delta_V.x(), preint->delta_V.y(), preint->delta_V.z());
        } else {
            LOG_WARN("  No preintegration for kf{} -> kf{}",
                    m_frame_window[i-1]->GetFrameId(), curr_kf->GetFrameId());
        }
    }
    LOG_INFO("IMU preintegration: {}/{} pairs ready", valid_preint_count, n_frames - 1);
    
    // Clear accumulated IMU data (not needed anymore for init)
    m_imu_since_last_keyframe.clear();
    
    // Add all keyframes to the list (first and last already added in caller)
    // Clear existing and re-add all in order
    m_keyframes.clear();
    m_all_keyframes.clear();
    for (int i = 0; i < n_frames; ++i) {
        m_keyframes.push_back(m_frame_window[i]);
        m_all_keyframes.push_back(m_frame_window[i]);
    }
    m_last_keyframe = m_frame_window.back();
    
    // Log MapPoint count for each keyframe
    for (int i = 0; i < n_frames; ++i) {
        auto& kf = m_frame_window[i];
        int mp_count = 0;
        size_t n_features = kf->GetFeatureCount();
        for (size_t j = 0; j < n_features; ++j) {
            auto mp = kf->GetMapPoint(j);
            if (mp && !mp->IsBad()) {
                mp_count++;
            }
        }
        LOG_INFO("KF[{}] frame_id={}: {} MapPoints (features={})", i, kf->GetFrameId(), mp_count, n_features);
    }
    
    LOG_INFO("Created {} keyframes from initialization window", n_frames);
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

bool Estimator::IsKeyframeInWindow(int frame_id) const {
    for (const auto& kf : m_keyframes) {
        if (kf->GetFrameId() == frame_id) return true;
    }
    return false;
}

bool Estimator::TriangulateSinglePoint(
    const Eigen::Vector3f& bearing1,
    const Eigen::Vector3f& bearing2,
    const Eigen::Matrix4f& T1w,
    const Eigen::Matrix4f& T2w,
    Eigen::Vector3f& point3d
) const {
    // Triangulation using SVD
    
    // Build 4x4 matrix A for SVD triangulation
    // A.row(0) = bearing_1(0) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(0);
    // A.row(1) = bearing_1(1) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(1);
    // A.row(2) = bearing_2(0) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(0);
    // A.row(3) = bearing_2(1) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(1);
    
    Eigen::Matrix4f A;
    A.row(0) = bearing1(0) * T1w.row(2) - bearing1(2) * T1w.row(0);
    A.row(1) = bearing1(1) * T1w.row(2) - bearing1(2) * T1w.row(1);
    A.row(2) = bearing2(0) * T2w.row(2) - bearing2(2) * T2w.row(0);
    A.row(3) = bearing2(1) * T2w.row(2) - bearing2(2) * T2w.row(1);
    
    // SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4f singular_vector = svd.matrixV().col(3);
    
    // Check for valid homogeneous coordinate
    if (std::abs(singular_vector(3)) < 1e-10f) {
        return false;
    }
    
    // Convert to 3D point
    point3d = singular_vector.head<3>() / singular_vector(3);
    
    // Equirectangular cameras skip depth check
    // because they can see in all directions (no "behind camera" concept)
    
    // Only check for valid (finite) coordinates and reasonable range
    if (!std::isfinite(point3d.x()) || !std::isfinite(point3d.y()) || !std::isfinite(point3d.z())) {
        return false;
    }
    
    // // Check reasonable depth range (distance from cameras)
    // Eigen::Vector3f cam1_center = -T1w.block<3,3>(0,0).transpose() * T1w.block<3,1>(0,3);
    // Eigen::Vector3f cam2_center = -T2w.block<3,3>(0,0).transpose() * T2w.block<3,1>(0,3);
    
    // float depth1 = (point3d - cam1_center).norm();
    // float depth2 = (point3d - cam2_center).norm();
    
    // constexpr float kMinDepth = 0.1f;
    // constexpr float kMaxDepth = 100.0f;
    
    // if (depth1 < kMinDepth || depth1 > kMaxDepth ||
    //     depth2 < kMinDepth || depth2 > kMaxDepth) {
    //     return false;
    // }
    
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
    
    LOG_DEBUG("Triangulation: kf1={} ({} feats), kf2={} ({} feats)",
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
    LOG_DEBUG("  Baseline: {:.6f}, C1=({:.4f},{:.4f},{:.4f}), C2=({:.4f},{:.4f},{:.4f})",
             baseline, C1.x(), C1.y(), C1.z(), C2.x(), C2.y(), C2.z());
    
    int triangulated_count = 0;
    int matched_count = 0;
    int already_has_mp = 0;
    int depth_failed = 0;
    int reproj_failed = 0;
    int parallax_failed = 0;
    
    
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
            depth_failed++;
            continue;
        }
        
        // Verify reprojection error before accepting the point
        // Transform point to both camera frames and check error
        Eigen::Vector4f point3d_h(point3d.x(), point3d.y(), point3d.z(), 1.0f);
        
        // Camera 1 reprojection
        Eigen::Vector4f pc1_h = T1w * point3d_h;
        Eigen::Vector3f pc1 = pc1_h.head<3>();
        Eigen::Vector3f reproj_bearing1 = pc1.normalized();
        float dot1 = bearing1.dot(reproj_bearing1);
        float angle_error1 = std::acos(std::min(1.0f, std::abs(dot1)));
        float pixel_error1 = angle_error1 * kf1->GetWidth() / (2.0f * M_PI);
        
        // Camera 2 reprojection
        Eigen::Vector4f pc2_h = T2w * point3d_h;
        Eigen::Vector3f pc2 = pc2_h.head<3>();
        Eigen::Vector3f reproj_bearing2 = pc2.normalized();
        float dot2 = bearing2.dot(reproj_bearing2);
        float angle_error2 = std::acos(std::min(1.0f, std::abs(dot2)));
        float pixel_error2 = angle_error2 * kf2->GetWidth() / (2.0f * M_PI);
        
        // Reject if reprojection error is too large
        // Chi-squared threshold: chi_sq_2D * sigma_sq = 5.99 * 1.0 ≈ 2.5 px
        // // For equirectangular with more noise, use relaxed threshold
        // const float max_reproj_error = 5.0f;
        // if (pixel_error1 > max_reproj_error || pixel_error2 > max_reproj_error) {
        //     reproj_failed++;
        //     continue;
        // }
        
        // Create new MapPoint
        auto mp = std::make_shared<MapPoint>(point3d);
        mp->SetTriangulated(true);
        
        // Set the earlier keyframe (kf1) as the reference keyframe for scale consistency
        mp->SetReferenceKeyframe(kf1);
        
        // Add observations to MapPoint
        mp->AddObservation(kf1, i1);
        mp->AddObservation(kf2, static_cast<int>(i2));
        
        // Register MapPoint to keyframes
        kf1->SetMapPoint(i1, mp);
        kf2->SetMapPoint(static_cast<int>(i2), mp);
        
        // Also add observations from intermediate frames in the sliding window
        // The feature in kf2 may have been tracked through multiple frames
        const auto& feat_observations = features2[i2]->GetObservations();
        for (const auto& obs : feat_observations) {
            auto obs_frame = obs.frame;
            if (!obs_frame) continue;
            
            // Skip if it's kf1 or kf2 (already added)
            if (obs_frame->GetFrameId() == kf1->GetFrameId() ||
                obs_frame->GetFrameId() == kf2->GetFrameId()) {
                continue;
            }
            
            // Only add if the frame is a keyframe in the current window
            if (!obs_frame->IsKeyframe()) continue;
            if (!IsKeyframeInWindow(obs_frame->GetFrameId())) continue;
            
            int obs_feat_idx = obs.feature_index;
            
            // Verify the observation is valid (check reprojection error)
            const auto& obs_features = obs_frame->GetFeatures();
            if (obs_feat_idx < 0 || obs_feat_idx >= static_cast<int>(obs_features.size())) {
                spdlog::error("Invalid feature index {} in frame {}", obs_feat_idx, obs_frame->GetFrameId());
                continue;
            }
            
            Eigen::Vector3f obs_bearing = obs_features[obs_feat_idx]->GetBearing();
            Eigen::Matrix4f T_obs_w = obs_frame->GetTwc().inverse();
            Eigen::Vector4f pt_h(point3d.x(), point3d.y(), point3d.z(), 1.0f);
            Eigen::Vector3f pt_cam = (T_obs_w * pt_h).head<3>();
            Eigen::Vector3f reproj_bearing = pt_cam.normalized();
            
            float dot = obs_bearing.dot(reproj_bearing);
            float angle_error = std::acos(std::min(1.0f, std::abs(dot)));
            float pixel_error = angle_error * obs_frame->GetWidth() / (2.0f * M_PI);
            
            // Only add if reprojection error is acceptable
            // if (pixel_error <= max_reproj_error) 
            {
                mp->AddObservation(obs_frame, obs_feat_idx);
                obs_frame->SetMapPoint(obs_feat_idx, mp);
            }
        }
        
        triangulated_count++;
    }
    
    LOG_DEBUG("  Matched: {}, already_has_mp: {}, depth_fail: {}, reproj_fail: {}, success: {}",
             matched_count, already_has_mp, depth_failed, reproj_failed, triangulated_count);
    
    return triangulated_count;
}

bool Estimator::TryInitializeIMU() {
    // Already initialized?
    if (m_imu_initialized) {
        return true;
    }
    
    // Need at least 3 keyframes for IMU initialization
    if (m_keyframes.size() < 3) {
        LOG_INFO("[IMU_INIT] Waiting for more keyframes: {}/3", m_keyframes.size());
        return false;
    }
    
    // Check if all keyframes have preintegration (except first)
    bool all_have_preint = true;
    for (size_t i = 1; i < m_keyframes.size(); ++i) {
        if (!m_keyframes[i]->HasIMUPreintegrationFromLastKeyframe()) {
            LOG_INFO("[IMU_INIT] Keyframe {} missing preintegration", i);
            all_have_preint = false;
            break;
        }
    }
    
    if (!all_have_preint) {
        LOG_INFO("[IMU_INIT] Not all keyframes have preintegration");
        return false;
    }
    
    LOG_INFO("[IMU_INIT] Starting IMU initialization with {} keyframes", m_keyframes.size());
    
    // Run 2-stage IMU optimization (gravity + scale + velocities + biases)
    Optimizer optimizer;
    auto result = optimizer.OptimizeIMUInit(m_keyframes);
    
    if (!result.success) {
        LOG_WARN("[IMU_INIT] IMU initialization failed");
        return false;
    }
    
    // Store results
    m_gravity = result.gravity;
    m_scale = result.scale;
    m_gyro_bias = result.gyro_bias;
    m_accel_bias = result.accel_bias;
    
    // Update velocities and biases in keyframes (before transformation)
    for (size_t i = 0; i < m_keyframes.size() && i < result.velocities.size(); ++i) {
        m_keyframes[i]->SetVelocity(result.velocities[i]);
        m_keyframes[i]->SetGyroBias(m_gyro_bias);
        m_keyframes[i]->SetAccelBias(m_accel_bias);
    }
    
    // Update biases in preintegrator
    m_imu_preintegrator->SetBias(m_gyro_bias, m_accel_bias);
    
    // Update current frame bias if exists
    if (m_current_frame) {
        m_current_frame->SetGyroBias(m_gyro_bias);
        m_current_frame->SetAccelBias(m_accel_bias);
    }
    
    LOG_INFO("[IMU_INIT] IMU initialization SUCCESS!");
    LOG_INFO("  Gravity (before transform): [{:.4f}, {:.4f}, {:.4f}] (norm={:.4f})",
             m_gravity.x(), m_gravity.y(), m_gravity.z(), m_gravity.norm());
    LOG_INFO("  Scale: {:.4f}", m_scale);
    LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}]",
             m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z());
    LOG_INFO("  Accel bias: [{:.6f}, {:.6f}, {:.6f}]",
             m_accel_bias.x(), m_accel_bias.y(), m_accel_bias.z());
    
    // Apply gravity alignment transformation to world coordinate system
    // This transforms all poses, MapPoints, and velocities so gravity points in -Z
    ApplyGravityAlignmentTransform(result.Rwg, static_cast<float>(result.scale));
    
    // Update gravity to point in -Z direction (after transformation)
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    
    m_imu_initialized = true;
    
    return true;
}

void Estimator::UpdatePreintegrationsWithNewBias(const Eigen::Vector3f& new_gyro_bias,
                                                  const Eigen::Vector3f& new_accel_bias) {
    // Update preintegrations in all keyframes
    for (auto& kf : m_keyframes) {
        auto preint = kf->GetIMUPreintegrationFromLastKeyframe();
        if (preint && preint->IsValid()) {
            m_imu_preintegrator->UpdatePreintegrationWithNewBias(preint, new_gyro_bias, new_accel_bias);
        }
        
        auto preint_frame = kf->GetIMUPreintegrationFromLastFrame();
        if (preint_frame && preint_frame->IsValid()) {
            m_imu_preintegrator->UpdatePreintegrationWithNewBias(preint_frame, new_gyro_bias, new_accel_bias);
        }
        
        // Update bias in frame
        kf->SetGyroBias(new_gyro_bias);
        kf->SetAccelBias(new_accel_bias);
    }
    
    // Update current frame if exists
    if (m_current_frame) {
        auto preint_frame = m_current_frame->GetIMUPreintegrationFromLastFrame();
        if (preint_frame && preint_frame->IsValid()) {
            m_imu_preintegrator->UpdatePreintegrationWithNewBias(preint_frame, new_gyro_bias, new_accel_bias);
        }
        m_current_frame->SetGyroBias(new_gyro_bias);
        m_current_frame->SetAccelBias(new_accel_bias);
    }
    
    // Update estimator's bias
    m_gyro_bias = new_gyro_bias;
    m_accel_bias = new_accel_bias;
    m_imu_preintegrator->SetBias(new_gyro_bias, new_accel_bias);
}

void Estimator::ApplyGravityAlignmentTransform(const Eigen::Matrix3f& Rwg, float optimized_scale) {
    // Tgw transforms from old world to gravity-aligned world
    // Rgw = Rwg^T (Rwg is world-to-gravity, we need gravity-to-world for inverse)
    Eigen::Matrix3f Rgw = Rwg.transpose();
    
    // Scale factor: optimized_scale means "VO * s = metric"
    // So to convert VO to metric, we apply 1/s
    float scale_factor = 1.0f / optimized_scale;
    
    // 1. Collect all unique MapPoints from keyframes
    std::set<std::shared_ptr<MapPoint>> unique_mps;
    for (const auto& kf : m_keyframes) {
        const auto& features = kf->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            auto mp = kf->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                unique_mps.insert(mp);
            }
        }
    }
    
    // =========================================================================
    // STEP 1: Apply gravity alignment (rotation only)
    // =========================================================================
    
    // Transform MapPoints: p_new = Rgw * p_old
    for (const auto& mp : unique_mps) {
        Eigen::Vector3f pos_old = mp->GetPosition();
        Eigen::Vector3f pos_new = Rgw * pos_old;
        mp->SetPosition(pos_new);
    }
    
    // Transform keyframe poses: R_new = Rgw * R_old, t_new = Rgw * t_old
    for (const auto& kf : m_keyframes) {
        Eigen::Matrix4f Twb_old = kf->GetTwb();
        Eigen::Matrix3f R_old = Twb_old.block<3,3>(0,0);
        Eigen::Vector3f t_old = Twb_old.block<3,1>(0,3);
        
        Eigen::Matrix4f Twb_new = Eigen::Matrix4f::Identity();
        Twb_new.block<3,3>(0,0) = Rgw * R_old;
        Twb_new.block<3,1>(0,3) = Rgw * t_old;
        
        kf->SetTwb(Twb_new);
    }
    
    // Transform velocities: v_new = Rgw * v_old
    for (const auto& kf : m_keyframes) {
        Eigen::Vector3f vel_old = kf->GetVelocity();
        Eigen::Vector3f vel_new = Rgw * vel_old;
        kf->SetVelocity(vel_new);
    }
    
    // =========================================================================
    // STEP 2: Apply scale correction (relative to first keyframe)
    // =========================================================================
    
    if (std::abs(scale_factor - 1.0f) > 1e-6) {
        // Scale keyframe poses relative to first keyframe
        Eigen::Matrix4f Twb_0 = m_keyframes[0]->GetTwb();
        
        for (size_t i = 1; i < m_keyframes.size(); ++i) {
            Eigen::Matrix4f Twb_i = m_keyframes[i]->GetTwb();
            Eigen::Matrix4f T_0_i = Twb_0.inverse() * Twb_i;
            T_0_i.block<3,1>(0,3) *= scale_factor;  // Scale translation relative to first
            m_keyframes[i]->SetTwb(Twb_0 * T_0_i);
        }
        
        // Scale velocities
        for (const auto& kf : m_keyframes) {
            Eigen::Vector3f vel = kf->GetVelocity();
            vel *= scale_factor;
            kf->SetVelocity(vel);
        }
        
        // Scale MapPoints relative to first camera
        Eigen::Matrix4f Twc_0 = m_keyframes[0]->GetTwc();
        Eigen::Matrix4f Tcw_0 = Twc_0.inverse();
        
        for (const auto& mp : unique_mps) {
            Eigen::Vector3f pos_w = mp->GetPosition();
            // Transform to first camera frame
            Eigen::Vector3f pos_c = Tcw_0.block<3,3>(0,0) * pos_w + Tcw_0.block<3,1>(0,3);
            // Apply scale
            pos_c *= scale_factor;
            // Transform back to world
            Eigen::Vector3f pos_w_new = Twc_0.block<3,3>(0,0) * pos_c + Twc_0.block<3,1>(0,3);
            mp->SetPosition(pos_w_new);
        }
    }
    
    // Check if current frame is already in keyframes (same object)
    bool current_in_keyframes = false;
    for (const auto& kf : m_keyframes) {
        if (kf.get() == m_current_frame.get()) {
            current_in_keyframes = true;
            break;
        }
    }
    
    // Update current pose from keyframe if it's the same object
    if (m_current_frame) {
        if (current_in_keyframes) {
            // Current frame is already transformed via keyframes, just update m_current_pose
            m_current_pose = m_current_frame->GetTwb();
        } else {
            // Current frame is different, need to transform separately
            Eigen::Matrix4f Twb_old = m_current_frame->GetTwb();
            Eigen::Matrix3f R_old = Twb_old.block<3,3>(0,0);
            Eigen::Vector3f t_old = Twb_old.block<3,1>(0,3);
            
            Eigen::Matrix4f Twb_new = Eigen::Matrix4f::Identity();
            Twb_new.block<3,3>(0,0) = Rgw * R_old;
            Twb_new.block<3,1>(0,3) = scale_factor * Rgw * t_old;
            
            m_current_frame->SetTwb(Twb_new);
            m_current_pose = Twb_new;
        }
    }
    
    LOG_INFO("Gravity aligned: scale={:.4f}, {} kfs, {} mps", 
             scale_factor, m_keyframes.size(), unique_mps.size());
}

} // namespace vio_360
