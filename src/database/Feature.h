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
#include <vector>
#include <memory>

namespace vio_360 {

// Forward declaration
class Frame;

/**
 * @brief Frame observation: which frame and which feature index
 */
struct FrameObservation {
    std::shared_ptr<Frame> frame;    // Frame that observed this feature
    int feature_index;               // Feature index in that frame
    
    FrameObservation(std::shared_ptr<Frame> f, int feat_idx) 
        : frame(f), feature_index(feat_idx) {}
};

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
    
    // ============ Frame Observation Management ============
    
    /**
     * @brief Add observation of this feature in a frame
     * @param frame Frame that observes this feature
     * @param feature_index Index of this feature in that frame
     */
    void AddObservation(std::shared_ptr<Frame> frame, int feature_index) {
        m_observations.emplace_back(frame, feature_index);
    }
    
    /**
     * @brief Get all observations
     */
    const std::vector<FrameObservation>& GetObservations() const { return m_observations; }
    
    /**
     * @brief Get number of observations
     */
    int GetObservationCount() const { return m_observations.size(); }
    
    /**
     * @brief Check if has observations
     */
    bool HasObservations() const { return !m_observations.empty(); }
    
    /**
     * @brief Clear all observations
     */
    void ClearObservations() { m_observations.clear(); }
    
    /**
     * @brief Update observations when tracking from previous feature
     * Copies all observations from tracked feature and adds current observation
     * @param prev_feature Previous feature that this feature tracks
     * @param current_frame Current frame
     * @param current_feature_index Index of this feature in current frame
     */
    void UpdateFeatureObservations(
        std::shared_ptr<Feature> prev_feature,
        std::shared_ptr<Frame> current_frame,
        int current_feature_index
    ) {
        if (prev_feature && prev_feature->HasObservations()) {
            // Copy all observations from previous feature
            m_observations = prev_feature->GetObservations();
        }
        // Add current observation
        AddObservation(current_frame, current_feature_index);
    }

private:
    int m_feature_id;              // Unique feature ID
    int m_tracked_feature_id;      // ID of the feature this tracks from previous frame (-1 if none)
    cv::Point2f m_pixel_coord;     // Pixel coordinates in image
    Eigen::Vector3f m_bearing;     // Unit bearing vector on sphere
    Eigen::Vector2f m_velocity;    // Optical flow velocity (pixels/frame)
    int m_track_count;             // Number of times successfully tracked
    int m_age;                     // Age since first detection (frames)
    bool m_is_valid;               // Whether this feature is valid
    
    // Frame observations: which frames observed this feature
    std::vector<FrameObservation> m_observations;
};

} // namespace vio_360
