/**
 * @file      Estimator.cpp
 * @brief     Main VIO estimator implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "FeatureTracker.h"
#include "Initializer.h"
#include "Camera.h"
#include "Frame.h"
#include "ConfigUtils.h"

#include <iostream>

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
    
    // Load initialization parameters from config
    m_window_size = config.initialization_window_size;
    m_min_parallax = config.initialization_min_parallax;
    m_frame_window.reserve(m_window_size);
    
    std::cout << "[ESTIMATOR] Initialized with camera " 
              << config.camera_width << "x" << config.camera_height << std::endl;
    std::cout << "[ESTIMATOR] Initialization: window_size=" << m_window_size
              << ", min_parallax=" << m_min_parallax << " pixels" << std::endl;
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
            TrackFeatures();
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
                std::cout << "[ESTIMATOR] System initialized successfully!" << std::endl;
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
        
        // TODO: Pose estimation (PnP + RANSAC)
        // TODO: Outlier rejection
        // TODO: Map point updates
        
        // Detect new features if needed
        if (m_current_frame->GetFeatureCount() < 100) {
            DetectFeatures();
        }
        
        // Check if should create keyframe
        if (ShouldCreateKeyframe()) {
            CreateKeyframe();
        }
        
        result.success = true;
        result.num_tracked = num_tracked;
    }
    
    // Update state
    result.pose = m_current_pose;
    result.num_features = m_current_frame->GetFeatureCount();
    m_all_frames.push_back(m_current_frame);
    m_previous_frame = m_current_frame;
    
    return result;
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
    
    // Sufficient parallax - print initialization attempt
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[ESTIMATOR] Initialization attempt!" << std::endl;
    std::cout << "  Window size: " << m_frame_window.size() << " frames" << std::endl;
    std::cout << "  Parallax: " << parallax << " pixels (min: " << m_min_parallax << ")" << std::endl;
    std::cout << "  Status: Ready to initialize!" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Step 1: Select features with sufficient observations
    auto selected_features = m_initializer->SelectFeaturesForInit(m_frame_window);
    
    if (selected_features.empty()) {
        std::cout << "[ESTIMATOR] Feature selection failed - cannot initialize" << std::endl;
        return false;
    }
    
    // Step 2: Try monocular initialization
    InitializationResult init_result;
    bool init_success = m_initializer->TryMonocularInitialization(m_frame_window, init_result);
    
    if (!init_success) {
        std::cout << "[ESTIMATOR] Monocular initialization failed" << std::endl;
        return false;
    }
    
    std::cout << "[ESTIMATOR] Monocular initialization succeeded!" << std::endl;
    
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
    
    std::cout << "[ESTIMATOR] Stored " << m_initialized_points.size() 
              << " 3D points and 2 camera poses" << std::endl;
    
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
    
    std::cout << "[ESTIMATOR] Reset complete" << std::endl;
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
    // TODO: Implement keyframe decision logic
    // 1. Check number of frames since last keyframe
    // 2. Check tracked feature ratio
    // 3. Check parallax/motion
    // 4. Check grid coverage
    
    // For now, create keyframe every 5 frames
    return (m_all_frames.size() % 5 == 0);
}

void Estimator::CreateKeyframe() {
    if (!m_current_frame) {
        return;
    }
    
    // Mark current frame as keyframe
    m_current_frame->SetKeyframe(true);
    m_keyframes.push_back(m_current_frame);
    m_last_keyframe = m_current_frame;
    
    // TODO: Triangulate new map points
    // TODO: Local bundle adjustment
    
    std::cout << "[ESTIMATOR] Created keyframe " << m_current_frame->GetFrameId() 
              << " (total: " << m_keyframes.size() << ")" << std::endl;
}

} // namespace vio_360
