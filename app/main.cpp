/**
 * @file      main.cpp
 * @brief     Feature tracking demo for 360-degree ERP images
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Camera.h"
#include "Feature.h"
#include "FeatureTracker.h"
#include "Frame.h"
#include "ConfigUtils.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

/**
 * @brief Get color based on tracking age (Blue -> Cyan -> Green -> Yellow -> Red)
 * @param age Tracking age (frames)
 * @param max_age Maximum age for color scaling (default 10)
 * @return BGR color
 */
cv::Scalar GetColorFromAge(int age, int max_age = 10) {
    age = std::min(age, max_age);
    float ratio = static_cast<float>(age) / static_cast<float>(max_age);
    
    int r, g, b;
    if (ratio < 0.25f) {
        // Blue -> Cyan
        r = 0;
        g = static_cast<int>(255 * (ratio / 0.25f));
        b = 255;
    } else if (ratio < 0.5f) {
        // Cyan -> Green
        r = 0;
        g = 255;
        b = static_cast<int>(255 * (1.0f - (ratio - 0.25f) / 0.25f));
    } else if (ratio < 0.75f) {
        // Green -> Yellow
        r = static_cast<int>(255 * ((ratio - 0.5f) / 0.25f));
        g = 255;
        b = 0;
    } else {
        // Yellow -> Red
        r = 255;
        g = static_cast<int>(255 * (1.0f - (ratio - 0.75f) / 0.25f));
        b = 0;
    }
    
    return cv::Scalar(b, g, r);  // OpenCV uses BGR
}

/**
 * @brief Visualize tracked features on image
 * @param image Input image
 * @param features Tracked features
 * @param show_tracks Whether to show tracking lines
 * @return Visualization image
 */
cv::Mat VisualizeFeatures(const cv::Mat& image, 
                         const std::vector<std::shared_ptr<vio_360::Feature>>& features,
                         const std::vector<std::shared_ptr<vio_360::Feature>>& prev_features,
                         bool show_tracks = true) {
    const auto& config = vio_360::ConfigUtils::GetInstance();
    
    cv::Mat vis_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, vis_image, cv::COLOR_GRAY2BGR);
    } else {
        vis_image = image.clone();
    }
    
    // Draw grid overlay if enabled
    if (config.visualization_show_grid) {
        const int grid_cols = config.grid_cols;
        const int grid_rows = config.grid_rows;
        const int img_width = vis_image.cols;
        const int img_height = vis_image.rows;
        const float cell_width = static_cast<float>(img_width) / grid_cols;
        const float cell_height = static_cast<float>(img_height) / grid_rows;
        
        // Compute feature clustering for each grid cell
        std::vector<std::vector<bool>> is_clustered(grid_rows, std::vector<bool>(grid_cols, false));
        
        if (config.visualization_highlight_clustered_grid && features.size() >= 2) {
            // Adaptive threshold based on grid cell size
            float grid_diagonal = std::sqrt(cell_width * cell_width + cell_height * cell_height);
            float std_threshold = grid_diagonal * config.visualization_clustered_std_ratio;
            
            // Assign features to grid cells and compute std
            for (int row = 0; row < grid_rows; ++row) {
                for (int col = 0; col < grid_cols; ++col) {
                    std::vector<cv::Point2f> cell_features;
                    
                    // Find features in this grid cell
                    for (const auto& feature : features) {
                        cv::Point2f pt = feature->GetPixelCoord();
                        int feat_col = std::min(static_cast<int>(pt.x / cell_width), grid_cols - 1);
                        int feat_row = std::min(static_cast<int>(pt.y / cell_height), grid_rows - 1);
                        
                        if (feat_row == row && feat_col == col) {
                            cell_features.push_back(pt);
                        }
                    }
                    
                    // Need at least 4 features to compute meaningful std
                    if (cell_features.size() >= 4) {
                        // Compute mean position
                        cv::Point2f mean(0.0f, 0.0f);
                        for (const auto& pt : cell_features) {
                            mean += pt;
                        }
                        mean.x /= cell_features.size();
                        mean.y /= cell_features.size();
                        
                        // Compute standard deviation
                        float variance = 0.0f;
                        for (const auto& pt : cell_features) {
                            float dx = pt.x - mean.x;
                            float dy = pt.y - mean.y;
                            variance += (dx * dx + dy * dy);
                        }
                        variance /= cell_features.size();
                        float std_dev = std::sqrt(variance);
                        
                        // Mark as clustered if std is below threshold
                        if (std_dev < std_threshold) {
                            is_clustered[row][col] = true;
                        }
                    }
                }
            }
        }
        
        // Draw grid cells with clustering overlay
        for (int row = 0; row < grid_rows; ++row) {
            for (int col = 0; col < grid_cols; ++col) {
                int x1 = static_cast<int>(col * cell_width);
                int y1 = static_cast<int>(row * cell_height);
                int x2 = static_cast<int>((col + 1) * cell_width);
                int y2 = static_cast<int>((row + 1) * cell_height);
                
                // Draw clustered grid overlay with transparency
                if (is_clustered[row][col]) {
                    cv::Mat roi = vis_image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                    cv::Mat overlay = roi.clone();
                    overlay.setTo(config.visualization_clustered_grid_color);
                    cv::addWeighted(overlay, 0.3, roi, 0.7, 0, roi);  // 30% overlay
                }
            }
        }
        
        // Draw vertical lines
        for (int i = 1; i < grid_cols; ++i) {
            int x = static_cast<int>(i * cell_width);
            cv::line(vis_image, 
                    cv::Point(x, 0), 
                    cv::Point(x, img_height),
                    config.visualization_grid_color,
                    config.visualization_grid_thickness);
        }
        
        // Draw horizontal lines
        for (int i = 1; i < grid_rows; ++i) {
            int y = static_cast<int>(i * cell_height);
            cv::line(vis_image,
                    cv::Point(0, y),
                    cv::Point(img_width, y),
                    config.visualization_grid_color,
                    config.visualization_grid_thickness);
        }
    }
    
    // Draw tracking lines if previous features available
    if (show_tracks && !prev_features.empty()) {
        for (const auto& feature : features) {
            if (feature->HasTrackedFeature()) {
                // Find corresponding previous feature
                for (const auto& prev_feature : prev_features) {
                    if (prev_feature->GetFeatureId() == feature->GetTrackedFeatureId()) {
                        cv::Point2f curr_pt = feature->GetPixelCoord();
                        cv::Point2f prev_pt = prev_feature->GetPixelCoord();
                        
                        cv::Scalar color = GetColorFromAge(feature->GetAge(), 
                                                          config.visualization_max_age_for_color);
                        cv::line(vis_image, prev_pt, curr_pt, color, 1, cv::LINE_AA);
                        break;
                    }
                }
            }
        }
    }
    
    // Draw feature points
    for (const auto& feature : features) {
        cv::Point2f pt = feature->GetPixelCoord();
        cv::Scalar color = GetColorFromAge(feature->GetAge(), 
                                          config.visualization_max_age_for_color);
        
        // Draw circle with age-based color
        cv::circle(vis_image, pt, 3, color, -1, cv::LINE_AA);
    }
    
    // Count stable features (age >= threshold)
    int stable_count = 0;
    for (const auto& feature : features) {
        if (feature->GetAge() >= config.visualization_stable_age_threshold) {
            stable_count++;
        }
    }
    
    // Add text information
    cv::putText(vis_image, "360 VIO - Feature Tracking", cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    std::string info = "Features: " + std::to_string(features.size());
    cv::putText(vis_image, info, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Add stable features count (top-right)
    std::string stable_text = "Stable (" + std::to_string(config.visualization_stable_age_threshold) + 
                             "+): " + std::to_string(stable_count);
    int text_width = cv::getTextSize(stable_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, nullptr).width;
    cv::putText(vis_image, stable_text, 
               cv::Point(vis_image.cols - text_width - 10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    
    // Add color legend
    int legend_y = 90;
    cv::putText(vis_image, "Age:", cv::Point(10, legend_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    for (int age : {0, 3, 5, 8, 10}) {
        cv::Scalar color = GetColorFromAge(age, config.visualization_max_age_for_color);
        int x = 60 + age * 20;
        cv::circle(vis_image, cv::Point(x, legend_y - 5), 5, color, -1);
        cv::putText(vis_image, std::to_string(age), cv::Point(x - 5, legend_y + 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    
    return vis_image;
}

/**
 * @brief Get sorted list of image files from directory
 */
std::vector<std::string> GetImageFiles(const std::string& directory) {
    std::vector<std::string> image_files;
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <images_directory> [output_video] [config_file]" << std::endl;
        std::cout << "Example: " << argv[0] << " ../datasets/seq_1/360-VIO_format/images output.mp4 ../config/default_config.yaml" << std::endl;
        return -1;
    }
    
    std::string images_dir = argv[1];
    std::string output_video = (argc > 2) ? argv[2] : "";
    std::string config_file = (argc > 3) ? argv[3] : "../config/default_config.yaml";
    
    // Load configuration
    auto& config = vio_360::ConfigUtils::GetInstance();
    if (!config.Load(config_file)) {
        std::cout << "Warning: Using default configuration values" << std::endl;
    }
    
    // Get image files
    std::vector<std::string> image_files = GetImageFiles(images_dir);
    if (image_files.empty()) {
        std::cerr << "Error: No images found in " << images_dir << std::endl;
        return -1;
    }
    
    std::cout << "Found " << image_files.size() << " images" << std::endl;
    
    // Read first image to get dimensions
    cv::Mat first_image = cv::imread(image_files[0], cv::IMREAD_GRAYSCALE);
    if (first_image.empty()) {
        std::cerr << "Error: Failed to read first image" << std::endl;
        return -1;
    }
    
    int original_width = first_image.cols;
    int original_height = first_image.rows;
    std::cout << "Original image size: " << original_width << "x" << original_height << std::endl;
    
    // Use configured tracking size
    int width = config.camera_width;
    int height = config.camera_height;
    std::cout << "Tracking image size: " << width << "x" << height << std::endl;
    
    // Create camera and feature tracker
    auto camera = std::make_shared<vio_360::Camera>(width, height);
    auto tracker = std::make_shared<vio_360::FeatureTracker>(
        camera,
        config.max_features,
        config.min_distance,
        config.quality_level
    );
    
    std::cout << "Initialized feature tracker" << std::endl;
    std::cout << "  Max features: " << config.max_features << std::endl;
    std::cout << "  Grid: " << config.grid_cols << "x" << config.grid_rows 
              << " (max " << config.max_features_per_grid << " per cell)" << std::endl;
    
    // Video writer setup
    cv::VideoWriter video_writer;
    if (!output_video.empty()) {
        int fourcc = cv::VideoWriter::fourcc(
            config.video_output_codec[0],
            config.video_output_codec[1],
            config.video_output_codec[2],
            config.video_output_codec[3]
        );
        
        video_writer.open(output_video, 
                         fourcc,
                         config.video_output_fps, 
                         cv::Size(width, height));
        
        if (!video_writer.isOpened()) {
            std::cerr << "Warning: Failed to open video writer" << std::endl;
        } else {
            std::cout << "Writing output video: " << output_video 
                     << " (" << width << "x" << height << ")" << std::endl;
        }
    }
    
    // Statistics
    std::vector<int> stable_feature_counts;
    
    // Process frames
    std::shared_ptr<vio_360::Frame> prev_frame = nullptr;
    int frame_id = 0;
    
    // Timing statistics
    std::vector<double> processing_times;
    std::vector<double> tracking_times;
    std::vector<double> visualization_times;
    std::vector<double> io_times;
    
    for (size_t i = 0; i < image_files.size(); ++i) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Read image
        auto io_start = std::chrono::high_resolution_clock::now();
        cv::Mat image = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Warning: Failed to read " << image_files[i] << std::endl;
            continue;
        }
        
        // Check if resize is needed
        cv::Mat working_image;
        if (image.cols != width || image.rows != height) {
            // Resize if image size doesn't match config
            cv::resize(image, working_image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        } else {
            // Use image as-is if already correct size
            working_image = image;
        }
        auto io_end = std::chrono::high_resolution_clock::now();
        
        // Create frame
        auto current_frame = std::make_shared<vio_360::Frame>(
            i * 33333333LL,  // timestamp (assuming 30fps)
            frame_id++,
            working_image,
            width,
            height
        );
        
        // Set grid parameters from config
        current_frame->SetGridParameters(
            config.grid_cols,
            config.grid_rows,
            config.max_features_per_grid
        );
        
        // Track features
        auto tracking_start = std::chrono::high_resolution_clock::now();
        tracker->TrackFeatures(current_frame, prev_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        
        // Get features from frame
        const auto& features = current_frame->GetFeatures();
        const auto& prev_features = prev_frame ? prev_frame->GetFeatures() : std::vector<std::shared_ptr<vio_360::Feature>>();
        
        // Visualize
        auto vis_start = std::chrono::high_resolution_clock::now();
        cv::Mat vis_image = VisualizeFeatures(working_image, features, prev_features, true);
        auto vis_end = std::chrono::high_resolution_clock::now();
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        // Calculate timing
        double io_time = std::chrono::duration<double, std::milli>(io_end - io_start).count();
        double tracking_time = std::chrono::duration<double, std::milli>(tracking_end - tracking_start).count();
        double vis_time = std::chrono::duration<double, std::milli>(vis_end - vis_start).count();
        double total_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        
        processing_times.push_back(total_time);
        tracking_times.push_back(tracking_time);
        visualization_times.push_back(vis_time);
        io_times.push_back(io_time);
        
        // Count stable features (using config threshold)
        int stable_count = 0;
        for (const auto& feature : features) {
            if (feature->GetAge() >= config.visualization_stable_age_threshold) {
                stable_count++;
            }
        }
        stable_feature_counts.push_back(stable_count);
        
        // Display
        cv::imshow("Feature Tracking", vis_image);
        
        // Write to video
        if (video_writer.isOpened()) {
            video_writer.write(vis_image);
        }
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == ' ') {  // Space: pause
            cv::waitKey(0);
        }
        
        // Print progress
        if ((i + 1) % 100 == 0) {
            int num_tracked, num_detected;
            tracker->GetTrackingStats(num_tracked, num_detected);
            
            std::cout << "Frame " << std::setw(4) << (i + 1) 
                     << " | Features: " << std::setw(4) << features.size()
                     << " | Tracked: " << std::setw(4) << num_tracked
                     << " | Detected: " << std::setw(4) << num_detected
                     << " | Stable (5+): " << std::setw(4) << stable_count;
            
            // Running average of stable features
            if (stable_feature_counts.size() > 30) {
                float avg = 0.0f;
                for (size_t j = stable_feature_counts.size() - 30; j < stable_feature_counts.size(); ++j) {
                    avg += stable_feature_counts[j];
                }
                avg /= 30.0f;
                std::cout << " | Avg: " << std::fixed << std::setprecision(1) << avg;
            }
            
            // Average processing time
            if (processing_times.size() > 30) {
                double avg_total = 0.0, avg_track = 0.0, avg_vis = 0.0, avg_io = 0.0;
                for (size_t j = processing_times.size() - 30; j < processing_times.size(); ++j) {
                    avg_total += processing_times[j];
                    avg_track += tracking_times[j];
                    avg_vis += visualization_times[j];
                    avg_io += io_times[j];
                }
                avg_total /= 30.0;
                avg_track /= 30.0;
                avg_vis /= 30.0;
                avg_io /= 30.0;
                
                double fps = 1000.0 / avg_total;
                std::cout << " | " << std::fixed << std::setprecision(1) << fps << " fps";
            }
            
            std::cout << std::endl;
        }
        
        prev_frame = current_frame;
    }
    
    // Cleanup
    cv::destroyAllWindows();
    if (video_writer.isOpened()) {
        video_writer.release();
    }
    
    // Print final statistics
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Processing Complete" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (stable_feature_counts.size() > 30) {
        int min_val = *std::min_element(stable_feature_counts.begin() + 30, 
                                       stable_feature_counts.end());
        int max_val = *std::max_element(stable_feature_counts.begin() + 30, 
                                       stable_feature_counts.end());
        
        float avg = 0.0f;
        for (size_t i = 30; i < stable_feature_counts.size(); ++i) {
            avg += stable_feature_counts[i];
        }
        avg /= (stable_feature_counts.size() - 30);
        
        std::cout << "\nStable Features (5+ frames, from frame 30):" << std::endl;
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Avg: " << std::fixed << std::setprecision(2) << avg << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
    }
    
    if (!output_video.empty() && video_writer.isOpened()) {
        std::cout << "\nVideo saved to: " << output_video << std::endl;
    }
    
    return 0;
}
