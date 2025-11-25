/**
 * @file      Feature.cpp
 * @brief     Implementation of Feature class.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Feature.h"

namespace vio_360 {

Feature::Feature(int feature_id, const cv::Point2f& pixel_coord)
    : m_feature_id(feature_id),
      m_tracked_feature_id(-1),
      m_pixel_coord(pixel_coord),
      m_bearing(Eigen::Vector3f::Zero()),
      m_velocity(Eigen::Vector2f::Zero()),
      m_track_count(0),
      m_age(0),
      m_is_valid(true) {
}

} // namespace vio_360
