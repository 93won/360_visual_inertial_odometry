/**
 * @file      MapPoint.h
 * @brief     3D map point representation for 360 VIO
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Dense>

namespace vio_360 {

// Forward declarations
class Frame;
class Feature;

/**
 * @brief Observation structure linking MapPoint to Frame
 */
struct Observation {
    std::weak_ptr<Frame> frame;     ///< Observing frame (weak to avoid cycles)
    int feature_index;               ///< Index of feature in frame
    
    Observation(std::shared_ptr<Frame> f, int idx)
        : frame(f), feature_index(idx) {}
};

/**
 * @brief 3D map point in world coordinates
 * 
 * Represents a triangulated 3D point observed by multiple frames.
 * Maintains position, observations, and validity flags.
 */
class MapPoint {
public:
    MapPoint();
    explicit MapPoint(const Eigen::Vector3f& position);
    ~MapPoint() = default;
    
    // ============ Position Management ============
    
    /**
     * @brief Set 3D position in world frame
     */
    void SetPosition(const Eigen::Vector3f& position);
    
    /**
     * @brief Get 3D position in world frame
     */
    Eigen::Vector3f GetPosition() const;
    
    // ============ Observation Management ============
    
    /**
     * @brief Add observation from a frame
     * @param frame Observing frame
     * @param feature_index Index of feature in frame
     */
    void AddObservation(std::shared_ptr<Frame> frame, int feature_index);
    
    /**
     * @brief Remove observation from a frame
     * @param frame Frame to remove
     */
    void RemoveObservation(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Get all observations
     */
    const std::vector<Observation>& GetObservations() const { return m_observations; }
    
    /**
     * @brief Get number of observations
     */
    int GetObservationCount() const { return static_cast<int>(m_observations.size()); }
    
    /**
     * @brief Check if observed by specific frame
     */
    bool IsObservedByFrame(std::shared_ptr<Frame> frame) const;
    
    /**
     * @brief Get feature index in specific frame
     * @return Feature index or -1 if not found
     */
    int GetFeatureIndexInFrame(std::shared_ptr<Frame> frame) const;
    
    // ============ ID Management ============
    
    int GetId() const { return m_id; }
    void SetId(int id) { m_id = id; }
    
    // ============ Status Flags ============
    
    void SetBad(bool bad = true) { m_is_bad = bad; }
    bool IsBad() const { return m_is_bad; }
    
    void SetTriangulated(bool triangulated = true) { m_is_triangulated = triangulated; }
    bool IsTriangulated() const { return m_is_triangulated; }
    
    void SetMarginalized(bool marginalized = true) { m_is_marginalized = marginalized; }
    bool IsMarginalized() const { return m_is_marginalized; }
    
    // ============ Reference Keyframe Management ============
    
    /**
     * @brief Set the reference (origin) keyframe for this MapPoint
     * The reference keyframe is the first keyframe that observed this point
     */
    void SetReferenceKeyframe(std::shared_ptr<Frame> frame) { m_reference_keyframe = frame; }
    
    /**
     * @brief Get the reference (origin) keyframe
     */
    std::shared_ptr<Frame> GetReferenceKeyframe() const { return m_reference_keyframe.lock(); }
    
    /**
     * @brief Check if given frame is the reference keyframe
     */
    bool IsReferenceKeyframe(std::shared_ptr<Frame> frame) const;
    
    // ============ Depth Management ============
    
    /**
     * @brief Get depth from specific frame's camera
     * @param frame Reference frame
     * @return Depth value or -1 if invalid
     */
    float GetDepthInFrame(std::shared_ptr<Frame> frame) const;
    
    /**
     * @brief Compute reprojection error in a frame
     * @param frame Reference frame
     * @return Reprojection error in pixels
     */
    float ComputeReprojectionError(std::shared_ptr<Frame> frame) const;

private:
    int m_id;                                    ///< Unique ID
    Eigen::Vector3f m_position;                  ///< 3D position in world frame
    std::vector<Observation> m_observations;     ///< List of observations
    
    bool m_is_bad;                               ///< Bad flag (invalid point)
    bool m_is_triangulated;                      ///< Successfully triangulated
    bool m_is_marginalized;                      ///< Marginalized flag (do not optimize or remove)
    
    std::weak_ptr<Frame> m_reference_keyframe;   ///< First keyframe that observed this point (origin)
    
    mutable std::mutex m_mutex;                  ///< Thread safety
    
    static int s_next_id;                        ///< ID counter
};

} // namespace vio_360
