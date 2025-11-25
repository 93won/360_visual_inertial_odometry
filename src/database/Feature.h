/**
 * @file      Feature.h
 * @brief     Defines the Feature class, representing a 2D feature in an ERP image.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace vio_360 {

/**
 * @brief Feature class representing a tracked 2D feature point
 */
class Feature {
public:
    Feature(int feature_id, const cv::Point2f& pixel_coord);
    ~Feature() = default;

    // ============ Getters ============
    
    int GetFeatureId() const { return m_feature_id; }
    cv::Point2f GetPixelCoord() const { return m_pixel_coord; }
    float GetU() const { return m_pixel_coord.x; }
    float GetV() const { return m_pixel_coord.y; }
    Eigen::Vector3f GetBearing() const { return m_bearing; }
    Eigen::Vector2f GetVelocity() const { return m_velocity; }
    int GetTrackCount() const { return m_track_count; }
    int GetAge() const { return m_age; }
    bool IsValid() const { return m_is_valid; }
    
    // ============ Setters ============
    
    void SetPixelCoord(const cv::Point2f& coord) { m_pixel_coord = coord; }
    void SetBearing(const Eigen::Vector3f& bearing) { m_bearing = bearing; }
    void SetVelocity(const Eigen::Vector2f& velocity) { m_velocity = velocity; }
    void SetTrackCount(int count) { m_track_count = count; }
    void SetAge(int age) { m_age = age; }
    void SetValid(bool valid) { m_is_valid = valid; }
    
    // ============ Tracking Relationship ============
    
    void SetTrackedFeatureId(int tracked_id) { m_tracked_feature_id = tracked_id; }
    int GetTrackedFeatureId() const { return m_tracked_feature_id; }
    bool HasTrackedFeature() const { return m_tracked_feature_id != -1; }
    
    // ============ Operations ============
    
    void IncrementTrackCount() { m_track_count++; }
    void IncrementAge() { m_age++; }

private:
    int m_feature_id;              // Unique feature ID
    int m_tracked_feature_id;      // ID of the feature this tracks from previous frame (-1 if none)
    cv::Point2f m_pixel_coord;     // Pixel coordinates in image
    Eigen::Vector3f m_bearing;     // Unit bearing vector on sphere
    Eigen::Vector2f m_velocity;    // Optical flow velocity (pixels/frame)
    int m_track_count;             // Number of times successfully tracked
    int m_age;                     // Age since first detection (frames)
    bool m_is_valid;               // Whether this feature is valid
};

} // namespace vio_360
