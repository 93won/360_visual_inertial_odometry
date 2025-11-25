/**
 * @file      Camera.cpp
 * @brief     Implementation of Camera class for Equirectangular camera model.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Camera.h"
#include <cmath>

namespace vio_360 {

Camera::Camera(int width, int height)
    : m_width(width), m_height(height) {
}

Eigen::Vector3f Camera::PixelToBearing(const cv::Point2f& pixel) const {
    // Normalize pixel coordinates
    float u_norm = pixel.x / static_cast<float>(m_width);   // [0, 1]
    float v_norm = pixel.y / static_cast<float>(m_height);  // [0, 1]
    
    // Convert to longitude and latitude
    float lon = u_norm * 2.0f * M_PI - M_PI;           // [-π, π]
    float lat = M_PI / 2.0f - v_norm * M_PI;           // [π/2, -π/2]
    
    // Spherical to Cartesian coordinates
    float cos_lat = std::cos(lat);
    Eigen::Vector3f bearing;
    bearing.x() = cos_lat * std::cos(lon);
    bearing.y() = cos_lat * std::sin(lon);
    bearing.z() = std::sin(lat);
    
    // Normalize (should already be unit vector, but ensure it)
    bearing.normalize();
    
    return bearing;
}

cv::Point2f Camera::BearingToPixel(const Eigen::Vector3f& bearing) const {
    // Cartesian to spherical coordinates
    float lon = std::atan2(bearing.y(), bearing.x());           // [-π, π]
    float lat = std::atan2(bearing.z(), 
                          std::sqrt(bearing.x() * bearing.x() + 
                                   bearing.y() * bearing.y()));  // [-π/2, π/2]
    
    // Convert to normalized pixel coordinates
    float u_norm = (lon + M_PI) / (2.0f * M_PI);     // [0, 1]
    float v_norm = (M_PI / 2.0f - lat) / M_PI;       // [0, 1]
    
    // Convert to pixel coordinates
    cv::Point2f pixel;
    pixel.x = u_norm * static_cast<float>(m_width);
    pixel.y = v_norm * static_cast<float>(m_height);
    
    return pixel;
}

cv::Point2f Camera::Project(const Eigen::Vector3f& point_c) const {
    // Normalize to get bearing vector
    Eigen::Vector3f bearing = point_c.normalized();
    
    // Convert bearing to pixel
    return BearingToPixel(bearing);
}

cv::Point2f Camera::ProjectWorld(const Eigen::Vector3f& point_w, const Eigen::Matrix4f& T_cw) const {
    // Transform point from world to camera frame
    Eigen::Vector4f point_w_homo;
    point_w_homo << point_w, 1.0f;
    
    Eigen::Vector4f point_c_homo = T_cw * point_w_homo;
    Eigen::Vector3f point_c = point_c_homo.head<3>();
    
    // Project to pixel
    return Project(point_c);
}

float Camera::AngularDistance(const Eigen::Vector3f& bearing1, 
                             const Eigen::Vector3f& bearing2) const {
    // Compute angular distance using dot product
    float cos_angle = bearing1.dot(bearing2);
    
    // Clamp to [-1, 1] to avoid numerical issues
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
    
    return std::acos(cos_angle);
}

cv::Mat Camera::CreatePolarMask(float top_ratio, float bottom_ratio) const {
    cv::Mat mask = cv::Mat::ones(m_height, m_width, CV_8U) * 255;
    
    // Calculate row indices for polar regions
    int top_rows = static_cast<int>(m_height * top_ratio);
    int bottom_start = static_cast<int>(m_height * (1.0f - bottom_ratio));
    
    // Mask out top polar region
    if (top_rows > 0) {
        mask(cv::Rect(0, 0, m_width, top_rows)) = 0;
    }
    
    // Mask out bottom polar region
    if (bottom_start < m_height) {
        mask(cv::Rect(0, bottom_start, m_width, m_height - bottom_start)) = 0;
    }
    
    return mask;
}

bool Camera::IsInPolarRegion(const cv::Point2f& pixel, float threshold) const {
    float v_ratio = pixel.y / static_cast<float>(m_height);
    return (v_ratio < threshold) || (v_ratio > (1.0f - threshold));
}

void Camera::WrapHorizontal(cv::Point2f& pixel) const {
    while (pixel.x < 0.0f) {
        pixel.x += static_cast<float>(m_width);
    }
    while (pixel.x >= static_cast<float>(m_width)) {
        pixel.x -= static_cast<float>(m_width);
    }
}

bool Camera::IsNearBoundary(const cv::Point2f& pixel, float margin) const {
    return (pixel.x < margin) || 
           (pixel.x > static_cast<float>(m_width) - margin) ||
           (pixel.y < margin) || 
           (pixel.y > static_cast<float>(m_height) - margin);
}

} // namespace vio_360
