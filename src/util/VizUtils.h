/**
 * @file      VizUtils.h
 * @brief     Visualization utilities for 360 VIO
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <mutex>

namespace vio_360 {

// Forward declarations
class Frame;
class Estimator;

/**
 * @brief Visualization utility class for 360 VIO
 */
class VizUtils {
public:
    /**
     * @brief Constructor
     * @param window_width Window width
     * @param window_height Window height
     */
    VizUtils(int window_width = 1920, int window_height = 1080);
    
    /**
     * @brief Destructor
     */
    ~VizUtils();
    
    /**
     * @brief Initialize viewer
     */
    void Initialize();
    
    /**
     * @brief Update visualization
     * @param estimator VIO estimator
     * @param current_frame Current frame being tracked
     * @param previous_frame Previous frame for optical flow visualization
     */
    void Update(const Estimator* estimator, 
                std::shared_ptr<Frame> current_frame,
                std::shared_ptr<Frame> previous_frame = nullptr);
    
    /**
     * @brief Check if viewer should close
     */
    bool ShouldClose();
    
    /**
     * @brief Get color from age
     */
    static cv::Scalar GetColorFromAge(int age, int max_age = 10);
    
    /**
     * @brief Draw tracking visualization
     */
    static cv::Mat DrawTracking(const cv::Mat& image,
                                std::shared_ptr<Frame> current_frame,
                                std::shared_ptr<Frame> previous_frame);

private:
    // Window dimensions
    int m_window_width;
    int m_window_height;
    
    // Pangolin display
    std::unique_ptr<pangolin::OpenGlRenderState> m_s_cam;
    std::unique_ptr<pangolin::View> m_d_cam;
    std::unique_ptr<pangolin::View> m_d_image;
    std::unique_ptr<pangolin::View> m_ui_panel;
    
    // Pangolin textures
    pangolin::GlTexture m_tracking_texture;
    
    // UI variables
    std::unique_ptr<pangolin::Var<bool>> m_show_trajectory;
    std::unique_ptr<pangolin::Var<bool>> m_show_keyframes;
    std::unique_ptr<pangolin::Var<bool>> m_show_map_points;
    std::unique_ptr<pangolin::Var<bool>> m_follow_camera;
    std::unique_ptr<pangolin::Var<int>> m_point_size;
    
    // Mutex for thread safety
    std::mutex m_mutex;
    
    /**
     * @brief Draw 3D scene
     */
    void Draw3DScene(const Estimator* estimator);
    
    /**
     * @brief Draw camera trajectory
     */
    void DrawTrajectory(const std::vector<std::shared_ptr<Frame>>& frames);
    
    /**
     * @brief Draw keyframes
     */
    void DrawKeyframes(const std::vector<std::shared_ptr<Frame>>& keyframes);
    
    /**
     * @brief Draw camera frustum
     */
    void DrawCamera(const Eigen::Matrix4f& T_wc, float r, float g, float b, float size = 0.1f);
    
    /**
     * @brief Draw coordinate axis
     */
    void DrawAxis(float size = 1.0f);
};

} // namespace vio_360
