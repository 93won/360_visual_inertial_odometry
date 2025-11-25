/**
 * @file      FeatureTracker.cpp
 * @brief     Implementation of FeatureTracker for ERP images.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "FeatureTracker.h"
#include "Frame.h"
#include "ConfigUtils.h"
#include <algorithm>
#include <random>
#include <cmath>

namespace vio_360 {

FeatureTracker::FeatureTracker(std::shared_ptr<Camera> camera,
                               int max_features,
                               float min_distance,
                               float quality_level)
    : m_camera(camera),
      m_max_features(max_features),
      m_min_distance(min_distance),
      m_quality_level(quality_level),
      m_lk_window_size(21, 21),
      m_lk_max_level(3),
      m_lk_criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01f),
      m_ransac_max_iterations(1000),
      m_ransac_threshold(2.0f),  // degrees
      m_ransac_confidence(0.99f),
      m_grid_cols(20),
      m_grid_rows(10),
      m_max_features_per_grid(4),
      m_next_feature_id(0),
      m_num_tracked(0),
      m_num_detected(0) {
    
    // Create polar region mask
    m_polar_mask = m_camera->CreatePolarMask();
}

void FeatureTracker::TrackFeatures(std::shared_ptr<Frame> current_frame,
                                   std::shared_ptr<Frame> previous_frame) {
    if (!current_frame) return;
    
    const cv::Mat& current_image = current_frame->GetImage();
    
    // First frame: detect features only
    if (!previous_frame || previous_frame->GetFeatures().empty()) {
        std::vector<cv::Point2f> detected_points = DetectNewFeatures(current_image);
        
        for (const auto& pt : detected_points) {
            auto feature = std::make_shared<Feature>(m_next_feature_id++, pt);
            feature->SetBearing(m_camera->PixelToBearing(pt));
            current_frame->AddFeature(feature);
        }
        
        m_num_tracked = 0;
        m_num_detected = static_cast<int>(detected_points.size());
        
        // Assign to grid and limit
        current_frame->AssignFeaturesToGrid();
        current_frame->LimitFeaturesPerGrid();
        
        m_prev_image = current_image.clone();
        return;
    }
    
    // Get previous features and image
    const auto& prev_features = previous_frame->GetFeatures();
    const cv::Mat& prev_image = previous_frame->GetImage();
    
    // Prepare previous points for tracking
    std::vector<cv::Point2f> prev_points;
    std::vector<std::shared_ptr<Feature>> prev_features_vec;
    
    for (const auto& feature : prev_features) {
        prev_points.push_back(feature->GetPixelCoord());
        prev_features_vec.push_back(feature);
    }
    
    // Track using optical flow
    std::vector<cv::Point2f> curr_points;
    std::vector<uchar> status;
    int num_tracked = TrackOpticalFlow(prev_image, current_image, 
                                       prev_points, curr_points, status);
    
    // Filter by status and polar regions
    std::vector<cv::Point2f> good_prev_points;
    std::vector<cv::Point2f> good_curr_points;
    std::vector<std::shared_ptr<Feature>> good_prev_features;
    
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && 
            !m_camera->IsInPolarRegion(curr_points[i]) &&
            !m_camera->IsNearBoundary(curr_points[i])) {
            
            good_prev_points.push_back(prev_points[i]);
            good_curr_points.push_back(curr_points[i]);
            good_prev_features.push_back(prev_features_vec[i]);
        }
    }
    
    // Reject outliers using rotation RANSAC
    std::vector<uchar> inlier_mask;
    if (good_curr_points.size() >= 3) {
        RejectOutliersRotationRANSAC(good_prev_points, good_curr_points, inlier_mask);
    } else {
        inlier_mask.resize(good_curr_points.size(), 1);
    }
    
    // Create features from inliers
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            auto prev_feature = good_prev_features[i];
            auto feature = std::make_shared<Feature>(prev_feature->GetFeatureId(), 
                                                     good_curr_points[i]);
            
            feature->SetBearing(m_camera->PixelToBearing(good_curr_points[i]));
            feature->SetTrackedFeatureId(prev_feature->GetFeatureId());
            feature->SetTrackCount(prev_feature->GetTrackCount() + 1);
            feature->SetAge(prev_feature->GetAge() + 1);
            
            // Compute velocity
            Eigen::Vector2f velocity;
            velocity.x() = good_curr_points[i].x - good_prev_points[i].x;
            velocity.y() = good_curr_points[i].y - good_prev_points[i].y;
            feature->SetVelocity(velocity);
            
            current_frame->AddFeature(feature);
        }
    }
    
    m_num_tracked = static_cast<int>(current_frame->GetFeatureCount());
    
    // Remove clustered features before grid assignment
    RemoveClusteredFeatures(current_frame);
    
    // Assign to grid and limit features per cell
    current_frame->AssignFeaturesToGrid();
    current_frame->LimitFeaturesPerGrid();
    
    // Detect new features if needed
    if (current_frame->GetFeatureCount() < static_cast<size_t>(m_max_features * 0.5f)) {
        cv::Mat mask = CreateFeatureMask(current_frame->GetFeatures());
        std::vector<cv::Point2f> new_points = DetectNewFeatures(current_image, mask);
        
        for (const auto& pt : new_points) {
            auto feature = std::make_shared<Feature>(m_next_feature_id++, pt);
            feature->SetBearing(m_camera->PixelToBearing(pt));
            current_frame->AddFeature(feature);
        }
        
        m_num_detected = static_cast<int>(new_points.size());
        
        // Reassign to grid and limit again
        current_frame->AssignFeaturesToGrid();
        current_frame->LimitFeaturesPerGrid();
    } else {
        m_num_detected = 0;
    }
    
    // Update state
    m_prev_image = current_image.clone();
}

std::vector<cv::Point2f> FeatureTracker::DetectNewFeatures(const cv::Mat& image, 
                                                           const cv::Mat& mask) {
    cv::Mat combined_mask;
    if (mask.empty()) {
        combined_mask = m_polar_mask.clone();
    } else {
        cv::bitwise_and(m_polar_mask, mask, combined_mask);
    }
    
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, m_max_features, m_quality_level,
                           m_min_distance, combined_mask, 3, false, 0.04);
    
    return corners;
}

int FeatureTracker::TrackOpticalFlow(const cv::Mat& prev_image,
                                     const cv::Mat& curr_image,
                                     const std::vector<cv::Point2f>& prev_points,
                                     std::vector<cv::Point2f>& curr_points,
                                     std::vector<uchar>& status) {
    if (prev_points.empty()) {
        return 0;
    }
    
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_image, curr_image, prev_points, curr_points,
                            status, error, m_lk_window_size, m_lk_max_level,
                            m_lk_criteria, 0, m_lk_criteria.epsilon);
    
    // Count successful tracks
    int num_tracked = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            num_tracked++;
        }
    }
    
    return num_tracked;
}

int FeatureTracker::RejectOutliersRotationRANSAC(
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<cv::Point2f>& curr_points,
    std::vector<uchar>& inlier_mask) {
    
    if (prev_points.size() < 3) {
        inlier_mask.resize(prev_points.size(), 1);
        return static_cast<int>(prev_points.size());
    }
    
    // Convert to bearing vectors
    std::vector<Eigen::Vector3f> prev_bearings;
    std::vector<Eigen::Vector3f> curr_bearings;
    
    for (size_t i = 0; i < prev_points.size(); ++i) {
        prev_bearings.push_back(m_camera->PixelToBearing(prev_points[i]));
        curr_bearings.push_back(m_camera->PixelToBearing(curr_points[i]));
    }
    
    // RANSAC
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(prev_points.size()) - 1);
    
    int best_inliers = 0;
    std::vector<uchar> best_mask;
    
    for (int iter = 0; iter < m_ransac_max_iterations; ++iter) {
        // Sample 3 points
        std::vector<int> indices;
        while (indices.size() < 3) {
            int idx = dis(gen);
            if (std::find(indices.begin(), indices.end(), idx) == indices.end()) {
                indices.push_back(idx);
            }
        }
        
        std::vector<Eigen::Vector3f> sample_prev;
        std::vector<Eigen::Vector3f> sample_curr;
        for (int idx : indices) {
            sample_prev.push_back(prev_bearings[idx]);
            sample_curr.push_back(curr_bearings[idx]);
        }
        
        // Estimate rotation
        Eigen::Matrix3f R = EstimateRotation(sample_prev, sample_curr);
        
        // Check for valid rotation
        if (std::abs(R.determinant() - 1.0f) > 0.1f) {
            continue;
        }
        
        // Count inliers
        float threshold_rad = m_ransac_threshold * M_PI / 180.0f;
        std::vector<uchar> mask = ComputeRotationInliers(prev_bearings, curr_bearings, 
                                                         R, threshold_rad);
        
        int num_inliers = 0;
        for (uchar m : mask) {
            if (m) num_inliers++;
        }
        
        if (num_inliers > best_inliers) {
            best_inliers = num_inliers;
            best_mask = mask;
        }
    }
    
    if (best_mask.empty()) {
        inlier_mask.resize(prev_points.size(), 1);
    } else {
        inlier_mask = best_mask;
    }
    
    return best_inliers;
}

Eigen::Matrix3f FeatureTracker::EstimateRotation(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2) {
    
    // Compute cross-covariance matrix
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < bearings1.size(); ++i) {
        H += bearings2[i] * bearings1[i].transpose();
    }
    
    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    // Rotation matrix
    Eigen::Matrix3f R = U * V.transpose();
    
    // Ensure proper rotation (det(R) = 1)
    if (R.determinant() < 0.0f) {
        V.col(2) *= -1.0f;
        R = U * V.transpose();
    }
    
    return R;
}

std::vector<uchar> FeatureTracker::ComputeRotationInliers(
    const std::vector<Eigen::Vector3f>& bearings1,
    const std::vector<Eigen::Vector3f>& bearings2,
    const Eigen::Matrix3f& rotation,
    float threshold) {
    
    std::vector<uchar> mask(bearings1.size(), 0);
    
    for (size_t i = 0; i < bearings1.size(); ++i) {
        // Rotate bearing1
        Eigen::Vector3f rotated = rotation * bearings1[i];
        
        // Compute angular distance
        float angular_error = m_camera->AngularDistance(rotated, bearings2[i]);
        
        // Check threshold
        if (angular_error < threshold) {
            mask[i] = 1;
        }
    }
    
    return mask;
}

void FeatureTracker::ManageGridFeatures(std::vector<std::shared_ptr<Feature>>& features) {
    // Grid-based feature management (simplified version)
    // Can be expanded with full grid logic from lightweight_vio if needed
}

cv::Mat FeatureTracker::CreateFeatureMask(
    const std::vector<std::shared_ptr<Feature>>& existing_features) {
    
    cv::Mat mask = m_polar_mask.clone();
    
    // Mask out regions around existing features
    for (const auto& feature : existing_features) {
        cv::Point2f pt = feature->GetPixelCoord();
        cv::circle(mask, pt, static_cast<int>(m_min_distance), 0, -1);
    }
    
    return mask;
}

void FeatureTracker::RemoveClusteredFeatures(std::shared_ptr<Frame> frame) {
    const auto& config = ConfigUtils::GetInstance();
    
    if (!config.visualization_highlight_clustered_grid) {
        return;  // Feature disabled
    }
    
    const auto& features = frame->GetFeatures();
    if (features.size() < 4) {
        return;  // Not enough features
    }
    
    // Get image dimensions and grid parameters
    const cv::Mat& image = frame->GetImage();
    const int img_width = image.cols;
    const int img_height = image.rows;
    const float cell_width = static_cast<float>(img_width) / m_grid_cols;
    const float cell_height = static_cast<float>(img_height) / m_grid_rows;
    const float grid_diagonal = std::sqrt(cell_width * cell_width + cell_height * cell_height);
    const float std_threshold = grid_diagonal * config.visualization_clustered_std_ratio;
    
    // Track which grid cells are clustered
    std::vector<std::vector<bool>> is_clustered(m_grid_rows, std::vector<bool>(m_grid_cols, false));
    std::vector<std::vector<std::vector<size_t>>> grid_feature_indices(
        m_grid_rows, std::vector<std::vector<size_t>>(m_grid_cols)
    );
    
    // Assign features to grid cells
    for (size_t idx = 0; idx < features.size(); ++idx) {
        const auto& feature = features[idx];
        cv::Point2f pt = feature->GetPixelCoord();
        int col = std::min(static_cast<int>(pt.x / cell_width), m_grid_cols - 1);
        int row = std::min(static_cast<int>(pt.y / cell_height), m_grid_rows - 1);
        grid_feature_indices[row][col].push_back(idx);
    }
    
    // Check each grid cell for clustering
    for (int row = 0; row < m_grid_rows; ++row) {
        for (int col = 0; col < m_grid_cols; ++col) {
            const auto& cell_indices = grid_feature_indices[row][col];
            
            // Need at least 4 features to compute meaningful std
            if (cell_indices.size() >= 4) {
                std::vector<cv::Point2f> cell_points;
                for (size_t idx : cell_indices) {
                    cell_points.push_back(features[idx]->GetPixelCoord());
                }
                
                // Compute mean position
                cv::Point2f mean(0.0f, 0.0f);
                for (const auto& pt : cell_points) {
                    mean += pt;
                }
                mean.x /= cell_points.size();
                mean.y /= cell_points.size();
                
                // Compute standard deviation
                float variance = 0.0f;
                for (const auto& pt : cell_points) {
                    float dx = pt.x - mean.x;
                    float dy = pt.y - mean.y;
                    variance += (dx * dx + dy * dy);
                }
                variance /= cell_points.size();
                float std_dev = std::sqrt(variance);
                
                // Mark as clustered if std is below threshold
                if (std_dev < std_threshold) {
                    is_clustered[row][col] = true;
                }
            }
        }
    }
    
    // Remove features from clustered grids
    std::vector<std::shared_ptr<Feature>> filtered_features;
    for (size_t idx = 0; idx < features.size(); ++idx) {
        const auto& feature = features[idx];
        cv::Point2f pt = feature->GetPixelCoord();
        int col = std::min(static_cast<int>(pt.x / cell_width), m_grid_cols - 1);
        int row = std::min(static_cast<int>(pt.y / cell_height), m_grid_rows - 1);
        
        // Keep only features NOT in clustered grids
        if (!is_clustered[row][col]) {
            filtered_features.push_back(feature);
        }
    }
    
    // Update frame with filtered features
    frame->ClearFeatures();
    for (const auto& feature : filtered_features) {
        frame->AddFeature(feature);
    }
}

void FeatureTracker::UpdateBearingVectors(std::vector<std::shared_ptr<Feature>>& features) {
    for (auto& feature : features) {
        Eigen::Vector3f bearing = m_camera->PixelToBearing(feature->GetPixelCoord());
        feature->SetBearing(bearing);
    }
}

void FeatureTracker::GetTrackingStats(int& num_tracked, int& num_detected) const {
    num_tracked = m_num_tracked;
    num_detected = m_num_detected;
}

} // namespace vio_360
