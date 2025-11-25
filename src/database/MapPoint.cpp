/**
 * @file      MapPoint.cpp
 * @brief     MapPoint implementation
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "MapPoint.h"
#include "Frame.h"
#include "Feature.h"

#include <algorithm>

namespace vio_360 {

int MapPoint::s_next_id = 0;

MapPoint::MapPoint()
    : m_id(s_next_id++)
    , m_position(Eigen::Vector3f::Zero())
    , m_is_bad(false)
    , m_is_triangulated(false)
{
}

MapPoint::MapPoint(const Eigen::Vector3f& position)
    : m_id(s_next_id++)
    , m_position(position)
    , m_is_bad(false)
    , m_is_triangulated(true)
{
}

void MapPoint::SetPosition(const Eigen::Vector3f& position) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_position = position;
}

Eigen::Vector3f MapPoint::GetPosition() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_position;
}

void MapPoint::AddObservation(std::shared_ptr<Frame> frame, int feature_index) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!frame) return;
    
    // Check if already observed by this frame
    for (auto& obs : m_observations) {
        if (auto existing_frame = obs.frame.lock()) {
            if (existing_frame == frame) {
                // Update feature index
                obs.feature_index = feature_index;
                return;
            }
        }
    }
    
    // Add new observation
    m_observations.emplace_back(frame, feature_index);
}

void MapPoint::RemoveObservation(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!frame) return;
    
    m_observations.erase(
        std::remove_if(m_observations.begin(), m_observations.end(),
            [&frame](const Observation& obs) {
                auto obs_frame = obs.frame.lock();
                return !obs_frame || obs_frame == frame;
            }),
        m_observations.end()
    );
    
    // Mark as bad if no observations left
    if (m_observations.empty()) {
        m_is_bad = true;
    }
}

bool MapPoint::IsObservedByFrame(std::shared_ptr<Frame> frame) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!frame) return false;
    
    for (const auto& obs : m_observations) {
        if (auto obs_frame = obs.frame.lock()) {
            if (obs_frame == frame) {
                return true;
            }
        }
    }
    return false;
}

int MapPoint::GetFeatureIndexInFrame(std::shared_ptr<Frame> frame) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!frame) return -1;
    
    for (const auto& obs : m_observations) {
        if (auto obs_frame = obs.frame.lock()) {
            if (obs_frame == frame) {
                return obs.feature_index;
            }
        }
    }
    return -1;
}

float MapPoint::GetDepthInFrame(std::shared_ptr<Frame> frame) const {
    if (!frame) return -1.0f;
    
    Eigen::Vector3f pos = GetPosition();
    
    // Transform to camera frame
    Eigen::Matrix4f T_wc = frame->GetTwc();
    Eigen::Matrix4f T_cw = T_wc.inverse();
    
    Eigen::Vector4f pos_h(pos.x(), pos.y(), pos.z(), 1.0f);
    Eigen::Vector3f pos_cam = (T_cw * pos_h).head<3>();
    
    return pos_cam.z();
}

float MapPoint::ComputeReprojectionError(std::shared_ptr<Frame> frame) const {
    if (!frame) return -1.0f;
    
    // Get feature index in this frame
    int feat_idx = GetFeatureIndexInFrame(frame);
    if (feat_idx < 0) return -1.0f;
    
    // Get observed feature
    const auto& features = frame->GetFeatures();
    if (feat_idx >= static_cast<int>(features.size())) return -1.0f;
    
    auto feature = features[feat_idx];
    if (!feature) return -1.0f;
    
    // Get observed bearing vector
    Eigen::Vector3f bearing_observed = feature->GetBearing();
    
    // Project 3D point to camera frame
    Eigen::Vector3f pos = GetPosition();
    Eigen::Matrix4f T_wc = frame->GetTwc();
    Eigen::Matrix4f T_cw = T_wc.inverse();
    
    Eigen::Vector4f pos_h(pos.x(), pos.y(), pos.z(), 1.0f);
    Eigen::Vector3f pos_cam = (T_cw * pos_h).head<3>();
    
    // Normalize to bearing vector
    Eigen::Vector3f bearing_projected = pos_cam.normalized();
    
    // Get image dimensions
    int width = frame->GetWidth();
    int height = frame->GetHeight();
    
    // Convert observed bearing to pixel coordinates (equirectangular projection)
    float lon_obs = std::atan2(bearing_observed.x(), bearing_observed.z());
    float lat_obs = -std::asin(bearing_observed.y());
    float u_obs = width * (0.5f + lon_obs / (2.0f * M_PI));
    float v_obs = height * (0.5f - lat_obs / M_PI);
    
    // Convert projected bearing to pixel coordinates
    float lon_proj = std::atan2(bearing_projected.x(), bearing_projected.z());
    float lat_proj = -std::asin(bearing_projected.y());
    float u_proj = width * (0.5f + lon_proj / (2.0f * M_PI));
    float v_proj = height * (0.5f - lat_proj / M_PI);
    
    // Handle equirectangular wraparound for longitude (±180 degrees)
    float du = u_obs - u_proj;
    if (du > width / 2.0f) {
        du -= width;
    } else if (du < -width / 2.0f) {
        du += width;
    }
    
    float dv = v_obs - v_proj;
    
    // Compute pixel error (Euclidean distance)
    return std::sqrt(du * du + dv * dv);
}

} // namespace vio_360
