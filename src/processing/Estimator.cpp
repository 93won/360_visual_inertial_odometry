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
    
    // TODO: Initialize monocular initializer (will be implemented later)
    // m_monocular_initializer = std::make_unique<MonocularInitializer>();
    
    std::cout << "[ESTIMATOR] Initialized with camera " 
              << config.camera_width << "x" << config.camera_height << std::endl;
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
        
        // Try to initialize
        bool init_success = TryInitialize();
        if (init_success) {
            std::cout << "[ESTIMATOR] Initialization successful!" << std::endl;
            m_initialized = true;
            result.success = true;
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
    // TODO: Implement monocular initialization
    // 1. Check if we have enough frames (e.g., 5-10 frames)
    // 2. Select best pair based on parallax
    // 3. Compute Essential matrix + RANSAC
    // 4. Recover pose (R, t)
    // 5. Triangulate initial map points
    // 6. Check reprojection error
    
    // For now, just return false
    return false;
}

void Estimator::Reset() {
    m_current_frame = nullptr;
    m_previous_frame = nullptr;
    m_last_keyframe = nullptr;
    m_all_frames.clear();
    m_keyframes.clear();
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
