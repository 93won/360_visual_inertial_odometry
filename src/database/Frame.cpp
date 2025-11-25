/**
 * @file      Frame.cpp
 * @brief     Implementation of Frame class for 360-degree VIO
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Frame.h"
#include "Feature.h"

namespace vio_360 {

Frame::Frame(long long timestamp, int frame_id, const cv::Mat& image, int width, int height)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_image(image.clone())
    , m_width(width)
    , m_height(height)
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_velocity(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gyro_bias(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_dt_from_last_keyframe(0.0)
    , m_T_BC(Eigen::Matrix4f::Identity())
    , m_T_CB(Eigen::Matrix4f::Identity())
    , m_grid_cols(20)
    , m_grid_rows(10)
    , m_max_features_per_grid(4)
{
    // Initialize grid
    m_feature_grid.resize(m_grid_rows, 
        std::vector<std::vector<int>>(m_grid_cols));
}

Eigen::Matrix4f Frame::GetTwb() const {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    
    Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
    T_wb.block<3, 3>(0, 0) = m_rotation;
    T_wb.block<3, 1>(0, 3) = m_translation;
    
    return T_wb;
}

void Frame::SetTwb(const Eigen::Matrix4f& T_wb) {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    
    m_rotation = T_wb.block<3, 3>(0, 0);
    m_translation = T_wb.block<3, 1>(0, 3);
}

Eigen::Matrix4f Frame::GetTwc() const {
    std::lock_guard<std::mutex> lock(m_pose_mutex);
    
    Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
    T_wb.block<3, 3>(0, 0) = m_rotation;
    T_wb.block<3, 1>(0, 3) = m_translation;
    
    // T_wc = T_wb * T_bc
    return T_wb * m_T_BC;
}

void Frame::SetTBC(const Eigen::Matrix4f& T_BC) {
    m_T_BC = T_BC;
    m_T_CB = T_BC.inverse();
}

// ============ Grid-based Feature Management ============

void Frame::SetGridParameters(int grid_cols, int grid_rows, int max_per_grid) {
    m_grid_cols = grid_cols;
    m_grid_rows = grid_rows;
    m_max_features_per_grid = max_per_grid;
    
    // Resize grid
    m_feature_grid.clear();
    m_feature_grid.resize(m_grid_rows, 
        std::vector<std::vector<int>>(m_grid_cols));
}

void Frame::AssignFeaturesToGrid() {
    // Clear grid
    for (auto& row : m_feature_grid) {
        for (auto& cell : row) {
            cell.clear();
        }
    }
    
    if (m_features.empty()) return;
    
    const float cell_width = static_cast<float>(m_width) / m_grid_cols;
    const float cell_height = static_cast<float>(m_height) / m_grid_rows;
    
    // Assign each feature to a grid cell
    for (size_t i = 0; i < m_features.size(); ++i) {
        const auto& feature = m_features[i];
        if (!feature) continue;
        
        cv::Point2f pixel = feature->GetPixelCoord();
        
        // Check bounds
        if (pixel.x < 0 || pixel.x >= m_width || 
            pixel.y < 0 || pixel.y >= m_height) {
            continue;
        }
        
        // Calculate grid coordinates
        int grid_x = std::min(static_cast<int>(pixel.x / cell_width), m_grid_cols - 1);
        int grid_y = std::min(static_cast<int>(pixel.y / cell_height), m_grid_rows - 1);
        
        // Add feature index to grid cell
        m_feature_grid[grid_y][grid_x].push_back(static_cast<int>(i));
    }
}

void Frame::LimitFeaturesPerGrid() {
    int total_removed = 0;
    
    // Process each grid cell
    for (int row = 0; row < m_grid_rows; ++row) {
        for (int col = 0; col < m_grid_cols; ++col) {
            auto& cell = m_feature_grid[row][col];
            
            // Skip if cell has fewer features than limit
            if (cell.size() <= static_cast<size_t>(m_max_features_per_grid)) {
                continue;
            }
            
            // Sort by track count (descending order)
            std::sort(cell.begin(), cell.end(), 
                [this](int a, int b) {
                    if (a >= m_features.size() || b >= m_features.size()) return false;
                    if (!m_features[a] || !m_features[b]) return false;
                    return m_features[a]->GetTrackCount() > m_features[b]->GetTrackCount();
                });
            
            // Mark excess features as invalid
            for (size_t i = m_max_features_per_grid; i < cell.size(); ++i) {
                int feature_idx = cell[i];
                if (feature_idx >= 0 && feature_idx < static_cast<int>(m_features.size())) {
                    // Don't actually remove, just mark for later filtering
                    total_removed++;
                }
            }
            
            // Keep only the top features in the cell
            cell.resize(m_max_features_per_grid);
        }
    }
    
    // Remove features that are not in any grid cell
    std::vector<bool> keep_feature(m_features.size(), false);
    for (const auto& row : m_feature_grid) {
        for (const auto& cell : row) {
            for (int idx : cell) {
                if (idx >= 0 && idx < static_cast<int>(m_features.size())) {
                    keep_feature[idx] = true;
                }
            }
        }
    }
    
    // Filter features
    std::vector<std::shared_ptr<Feature>> filtered_features;
    for (size_t i = 0; i < m_features.size(); ++i) {
        if (keep_feature[i]) {
            filtered_features.push_back(m_features[i]);
        }
    }
    
    m_features = std::move(filtered_features);
    
    // Reassign to grid with new indices
    AssignFeaturesToGrid();
}

std::vector<int> Frame::SelectFeaturesForTracking() const {
    std::vector<int> selected_indices;
    
    // Select best feature from each grid cell
    for (const auto& row : m_feature_grid) {
        for (const auto& cell : row) {
            if (cell.empty()) continue;
            
            // Find feature with highest track count
            int best_idx = -1;
            int max_track_count = -1;
            
            for (int idx : cell) {
                if (idx >= 0 && idx < static_cast<int>(m_features.size()) && m_features[idx]) {
                    int track_count = m_features[idx]->GetTrackCount();
                    if (track_count > max_track_count) {
                        max_track_count = track_count;
                        best_idx = idx;
                    }
                }
            }
            
            if (best_idx >= 0) {
                selected_indices.push_back(best_idx);
            }
        }
    }
    
    return selected_indices;
}

} // namespace vio_360
