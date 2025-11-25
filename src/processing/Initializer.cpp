/**
 * @file      Initializer.cpp
 * @brief     Implementation of Initializer
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Initializer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace vio_360 {

Initializer::Initializer()
    : m_is_initialized(false)
    , m_min_features(50)
    , m_min_observations(10)
    , m_min_parallax(10.0f)
    , m_ransac_threshold(0.001f)
    , m_ransac_iterations(200)
    , m_min_inlier_ratio(0.7f)
    , m_max_reprojection_error(2.0f)
{
}

void Initializer::Reset() {
    m_is_initialized = false;
}

bool Initializer::TryMonocularInitialization(
    const std::vector<std::shared_ptr<Frame>>& frames
) {
    // TODO: Implement full initialization pipeline
    return false;
}

float Initializer::ComputeParallax(
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
    
    // Find correspondences between two frames
    // Match by feature ID (tracked features have same ID)
    std::vector<float> parallaxes;
    parallaxes.reserve(features1.size());
    
    for (const auto& feat1 : features1) {
        // Find matching feature in frame2 by ID
        for (const auto& feat2 : features2) {
            if (feat1->GetFeatureId() == feat2->GetFeatureId()) {
                // Found correspondence
                cv::Point2f pt1 = feat1->GetPixelCoord();
                cv::Point2f pt2 = feat2->GetPixelCoord();
                
                // Compute Euclidean distance (parallax in pixels)
                float dx = pt2.x - pt1.x;
                float dy = pt2.y - pt1.y;
                float parallax = std::sqrt(dx * dx + dy * dy);
                
                parallaxes.push_back(parallax);
                break;
            }
        }
    }
    
    if (parallaxes.empty()) {
        return 0.0f;
    }
    
    // Compute median parallax (more robust than mean)
    std::sort(parallaxes.begin(), parallaxes.end());
    
    size_t mid = parallaxes.size() / 2;
    float median_parallax;
    
    if (parallaxes.size() % 2 == 0) {
        median_parallax = (parallaxes[mid - 1] + parallaxes[mid]) / 2.0f;
    } else {
        median_parallax = parallaxes[mid];
    }
    
    return median_parallax;
}

std::vector<std::shared_ptr<Feature>> Initializer::SelectFeaturesForInit(
    const std::vector<std::shared_ptr<Frame>>& frames
) const {
    // TODO: Implement feature selection
    std::vector<std::shared_ptr<Feature>> selected_features;
    return selected_features;
}

bool Initializer::SelectFramePair(
    const std::vector<std::shared_ptr<Feature>>& features,
    std::shared_ptr<Frame>& frame1,
    std::shared_ptr<Frame>& frame2
) const {
    // TODO: Implement frame pair selection
    return false;
}

bool Initializer::ComputeEssentialMatrix(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    Eigen::Matrix3f& E,
    std::vector<bool>& inlier_mask
) const {
    // TODO: Implement Essential matrix computation with RANSAC
    return false;
}

bool Initializer::RecoverPose(
    const Eigen::Matrix3f& E,
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t,
    const std::vector<bool>& inlier_mask
) const {
    // TODO: Implement pose recovery from Essential matrix
    return false;
}

int Initializer::TriangulatePoints(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t,
    std::vector<Eigen::Vector3f>& points3d
) const {
    // TODO: Implement triangulation
    return 0;
}

} // namespace vio_360
