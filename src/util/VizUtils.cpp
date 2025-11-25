/**
 * @file      VizUtils.cpp
 * @brief     Visualization utilities implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "VizUtils.h"
#include "Estimator.h"
#include "Frame.h"
#include "Feature.h"
#include <iostream>

namespace vio_360 {

VizUtils::VizUtils(int window_width, int window_height)
    : m_window_width(window_width)
    , m_window_height(window_height) {
}

VizUtils::~VizUtils() {
}

void VizUtils::Initialize() {
    // Create Pangolin window
    pangolin::CreateWindowAndBind("360 VIO Viewer", m_window_width, m_window_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Define Projection and initial ModelView matrix
    m_s_cam = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(m_window_width, m_window_height, 500, 500, 
                                   m_window_width/2, m_window_height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin::AxisZ)
    );
    
    // Create Interactive View in window
    const int UI_WIDTH = 640;
    const int IMAGE_HEIGHT = 360;
    
    // UI Panel (left side, top part)
    m_ui_panel = std::make_unique<pangolin::View>();
    m_ui_panel->SetBounds(pangolin::Attach::Pix(IMAGE_HEIGHT), 1.0, 
                          0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::DisplayBase().AddDisplay(*m_ui_panel);
    
    // 3D View (right side, full height)
    m_d_cam = std::make_unique<pangolin::View>();
    m_d_cam->SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
           .SetHandler(new pangolin::Handler3D(*m_s_cam));
    pangolin::DisplayBase().AddDisplay(*m_d_cam);
    
    // Image View (left side bottom, 640x360)
    m_d_image = std::make_unique<pangolin::View>();
    m_d_image->SetBounds(0.0, pangolin::Attach::Pix(IMAGE_HEIGHT),
                         0.0, pangolin::Attach::Pix(UI_WIDTH))
             .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    pangolin::DisplayBase().AddDisplay(*m_d_image);
    
    // Create UI Panel
    pangolin::CreatePanel("ui")
        .SetBounds(pangolin::Attach::Pix(IMAGE_HEIGHT), 1.0, 
                   0.0, pangolin::Attach::Pix(UI_WIDTH));
    
    // UI Variables
    m_show_trajectory = std::make_unique<pangolin::Var<bool>>("ui.Show Trajectory", true, true);
    m_show_keyframes = std::make_unique<pangolin::Var<bool>>("ui.Show Keyframes", true, true);
    m_show_map_points = std::make_unique<pangolin::Var<bool>>("ui.Show Map Points", false, true);
    m_follow_camera = std::make_unique<pangolin::Var<bool>>("ui.Follow Camera", true, true);
    m_point_size = std::make_unique<pangolin::Var<int>>("ui.Point Size", 2, 1, 10);
    
    std::cout << "[VIZUTILS] Initialized" << std::endl;
}

void VizUtils::Update(const Estimator* estimator, 
                      std::shared_ptr<Frame> current_frame,
                      std::shared_ptr<Frame> previous_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Clear screen with black background
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Black background for UI panel
    
    // Activate 3D view with dark navy background
    m_d_cam->Activate(*m_s_cam);
    glClearColor(0.05f, 0.05f, 0.15f, 1.0f);  // Dark navy for 3D scene
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Draw 3D scene
    Draw3DScene(estimator);
    
    // Draw tracking visualization
    if (current_frame && !current_frame->GetImage().empty()) {
        // Create tracking visualization
        cv::Mat tracking_image = DrawTracking(current_frame->GetImage(), 
                                              current_frame, 
                                              previous_frame);
        
        cv::Mat rgb_image;
        if (tracking_image.channels() == 1) {
            cv::cvtColor(tracking_image, rgb_image, cv::COLOR_GRAY2RGB);
        } else if (tracking_image.channels() == 3) {
            cv::cvtColor(tracking_image, rgb_image, cv::COLOR_BGR2RGB);
        } else {
            rgb_image = tracking_image;
        }
        
        // Resize to fixed size (640x360)
        cv::Mat resized_image;
        cv::resize(rgb_image, resized_image, cv::Size(640, 360));
        
        // Flip vertically for OpenGL (OpenGL origin is bottom-left, OpenCV is top-left)
        cv::Mat flipped_image;
        cv::flip(resized_image, flipped_image, 0);
        
        // Upload to texture
        if (!m_tracking_texture.IsValid() || 
            m_tracking_texture.width != flipped_image.cols || 
            m_tracking_texture.height != flipped_image.rows) {
            m_tracking_texture.Reinitialise(flipped_image.cols, flipped_image.rows, 
                                           GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
        }
        m_tracking_texture.Upload(flipped_image.data, GL_RGB, GL_UNSIGNED_BYTE);
        
        // Display image
        m_d_image->Activate();
        glColor3f(1.0f, 1.0f, 1.0f);
        m_tracking_texture.RenderToViewport();
    }
    
    // Swap buffers
    pangolin::FinishFrame();
}

bool VizUtils::ShouldClose() {
    return pangolin::ShouldQuit();
}

void VizUtils::Draw3DScene(const Estimator* estimator) {
    if (!estimator) return;
    
    // Draw coordinate axis
    DrawAxis(1.0f);
    
    // Draw trajectory
    if (*m_show_trajectory) {
        const auto& frames = estimator->GetAllFrames();
        DrawTrajectory(frames);
    }
    
    // Draw keyframes
    if (*m_show_keyframes) {
        // TODO: Get keyframes from estimator
        // DrawKeyframes(keyframes);
    }
    
    // Draw current camera
    auto current_frame = estimator->GetCurrentFrame();
    if (current_frame) {
        Eigen::Matrix4f T_wc = current_frame->GetTwc();
        DrawCamera(T_wc, 0.0f, 1.0f, 0.0f, 0.1f);  // Green for current
    }
    
    // Follow camera if enabled
    if (*m_follow_camera && current_frame) {
        Eigen::Matrix4f T_wc = current_frame->GetTwc();
        Eigen::Vector3f pos = T_wc.block<3, 1>(0, 3);
        
        m_s_cam->Follow(pangolin::OpenGlMatrix::Translate(pos.x(), pos.y(), pos.z()));
    }
}

void VizUtils::DrawTrajectory(const std::vector<std::shared_ptr<Frame>>& frames) {
    if (frames.size() < 2) return;
    
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.0f, 0.0f);  // Red trajectory
    glBegin(GL_LINE_STRIP);
    
    for (const auto& frame : frames) {
        Eigen::Matrix4f T_wc = frame->GetTwc();
        Eigen::Vector3f pos = T_wc.block<3, 1>(0, 3);
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    
    glEnd();
}

void VizUtils::DrawKeyframes(const std::vector<std::shared_ptr<Frame>>& keyframes) {
    for (const auto& kf : keyframes) {
        Eigen::Matrix4f T_wc = kf->GetTwc();
        DrawCamera(T_wc, 0.0f, 0.0f, 1.0f, 0.05f);  // Blue for keyframes
    }
}

void VizUtils::DrawCamera(const Eigen::Matrix4f& T_wc, float r, float g, float b, float size) {
    const float w = size;
    const float h = w * 0.75f;
    const float z = w * 0.6f;
    
    glPushMatrix();
    
    // Apply transformation
    Eigen::Matrix4f T_cw = T_wc.inverse();
    glMultMatrixf(T_cw.data());
    
    glLineWidth(2.0f);
    glColor3f(r, g, b);
    
    // Draw camera frustum
    glBegin(GL_LINES);
    
    // Camera center to corners
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);
    
    // Image plane rectangle
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    
    glVertex3f(w, -h, z);
    glVertex3f(-w, -h, z);
    
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    
    glEnd();
    
    glPopMatrix();
}

void VizUtils::DrawAxis(float size) {
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(size, 0, 0);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, size, 0);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, size);
    
    glEnd();
}

cv::Scalar VizUtils::GetColorFromAge(int age, int max_age) {
    age = std::min(age, max_age);
    float ratio = static_cast<float>(age) / static_cast<float>(max_age);
    
    int r, g, b;
    if (ratio < 0.25f) {
        r = 0;
        g = static_cast<int>(255 * (ratio / 0.25f));
        b = 255;
    } else if (ratio < 0.5f) {
        r = 0;
        g = 255;
        b = static_cast<int>(255 * (1.0f - (ratio - 0.25f) / 0.25f));
    } else if (ratio < 0.75f) {
        r = static_cast<int>(255 * ((ratio - 0.5f) / 0.25f));
        g = 255;
        b = 0;
    } else {
        r = 255;
        g = static_cast<int>(255 * (1.0f - (ratio - 0.75f) / 0.25f));
        b = 0;
    }
    
    return cv::Scalar(b, g, r);
}

cv::Mat VizUtils::DrawTracking(const cv::Mat& image,
                               std::shared_ptr<Frame> current_frame,
                               std::shared_ptr<Frame> previous_frame) {
    cv::Mat vis_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, vis_image, cv::COLOR_GRAY2BGR);
    } else {
        vis_image = image.clone();
    }
    
    if (!current_frame) return vis_image;
    
    const auto& features = current_frame->GetFeatures();
    
    // Draw tracking lines
    if (previous_frame) {
        const auto& prev_features = previous_frame->GetFeatures();
        
        for (const auto& feature : features) {
            if (feature->HasTrackedFeature()) {
                for (const auto& prev_feature : prev_features) {
                    if (prev_feature->GetFeatureId() == feature->GetTrackedFeatureId()) {
                        cv::Point2f curr_pt = feature->GetPixelCoord();
                        cv::Point2f prev_pt = prev_feature->GetPixelCoord();
                        
                        cv::Scalar color = GetColorFromAge(feature->GetAge(), 10);
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
        cv::Scalar color = GetColorFromAge(feature->GetAge(), 10);
        cv::circle(vis_image, pt, 3, color, -1, cv::LINE_AA);
    }
    
    // Add text information
    cv::putText(vis_image, "360 VIO Tracking", cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    std::string info = "Features: " + std::to_string(features.size());
    cv::putText(vis_image, info, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    if (current_frame->IsKeyframe()) {
        cv::putText(vis_image, "KEYFRAME", cv::Point(10, 90),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }
    
    return vis_image;
}

} // namespace vio_360
