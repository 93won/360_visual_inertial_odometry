/**
 * @file      VizUtils.cpp
 * @brief     Visualization utilities implementation
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
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
#include "MapPoint.h"
#include <iostream>
#include <GL/glu.h>
#include <set>

namespace vio_360 {

VizUtils::VizUtils(int window_width, int window_height)
    : m_window_width(window_width)
    , m_window_height(window_height)
    , m_paused(false)
    , m_step_requested(false) {
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
    // 3D view will be on the right side (1920-640 = 1280 width, 1080 height)
    const int UI_WIDTH = 640;
    const int view_width = m_window_width - UI_WIDTH;   // 1280
    const int view_height = m_window_height;             // 1080
    
    m_s_cam = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(view_width, view_height, 500, 500, 
                                   view_width/2, view_height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin::AxisZ)
    );
    
    // Create Interactive View in window
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
    m_show_init_result = std::make_unique<pangolin::Var<bool>>("ui.Show Init Result", true, true);
    m_follow_camera = std::make_unique<pangolin::Var<bool>>("ui.Follow Camera", true, true);
    m_point_size = std::make_unique<pangolin::Var<int>>("ui.Point Size", 2, 1, 10);
    m_pause_button = std::make_unique<pangolin::Var<bool>>("ui.Pause", false, false);
    m_step_button = std::make_unique<pangolin::Var<bool>>("ui.Step", false, false);
}

void VizUtils::Update(const Estimator* estimator, 
                      std::shared_ptr<Frame> current_frame,
                      std::shared_ptr<Frame> previous_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Check pause button
    if (pangolin::Pushed(*m_pause_button)) {
        m_paused = !m_paused;
    }
    
    // Check step button (only works when paused)
    if (pangolin::Pushed(*m_step_button)) {
        if (m_paused) {
            m_step_requested = true;
        }
    }
    
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
    
    // Draw MapPoints from all keyframes
    if (*m_show_init_result && estimator->IsInitialized()) {
        const auto& keyframes = estimator->GetKeyframes();
        
        // Collect unique MapPoints from all keyframes
        std::set<std::shared_ptr<MapPoint>> unique_mappoints;
        for (const auto& kf : keyframes) {
            const auto& features = kf->GetFeatures();
            for (size_t i = 0; i < features.size(); ++i) {
                auto mp = kf->GetMapPoint(static_cast<int>(i));
                if (mp && !mp->IsBad()) {
                    unique_mappoints.insert(mp);
                }
            }
        }
        
        // Draw MapPoints
        if (!unique_mappoints.empty()) {
            glPointSize(static_cast<float>(*m_point_size) * 2.0f);
            glBegin(GL_POINTS);
            glColor3f(0.0f, 1.0f, 0.0f);  // Green for MapPoints
            
            for (const auto& mp : unique_mappoints) {
                Eigen::Vector3f pt = mp->GetPosition();
                glVertex3f(pt.x(), pt.y(), pt.z());
            }
            
            glEnd();
        }
    }
    
    // Draw trajectory
    if (*m_show_trajectory) {
        const auto& frames = estimator->GetAllFrames();
        DrawTrajectory(frames);
    }
    
    // Draw keyframes as spheres (only window keyframes)
    if (*m_show_keyframes && estimator->IsInitialized()) {
        const auto& keyframes = estimator->GetKeyframes();  // Only active window keyframes
        for (size_t i = 0; i < keyframes.size(); ++i) {
            const auto& kf = keyframes[i];
            
            Eigen::Matrix4f T_wb = kf->GetTwb();
            Eigen::Vector3f pos = T_wb.block<3, 1>(0, 3);
            
            glColor3f(0.0f, 0.9f, 0.7f);  // Mint color for window keyframes
            
            // Draw wireframe sphere
            glPushMatrix();
            glTranslatef(pos.x(), pos.y(), pos.z());
            GLUquadric* quad = gluNewQuadric();
            gluQuadricDrawStyle(quad, GLU_LINE);  // Wireframe style
            gluSphere(quad, 0.15, 12, 8);  // radius=0.15 (10x larger), slices=12, stacks=8
            gluDeleteQuadric(quad);
            glPopMatrix();
        }
    }
    
    // Follow camera if enabled
    auto current_frame = estimator->GetCurrentFrame();
    if (*m_follow_camera && current_frame && estimator->IsInitialized()) {
        Eigen::Matrix4f T_wb = current_frame->GetTwb();
        Eigen::Vector3f pos = T_wb.block<3, 1>(0, 3);
        
        m_s_cam->Follow(pangolin::OpenGlMatrix::Translate(pos.x(), pos.y(), pos.z()));
    }
    
    // Draw current frame position as RED wireframe sphere (tracking indicator)
    if (current_frame && estimator->IsInitialized()) {
        Eigen::Matrix4f T_wb = current_frame->GetTwb();
        Eigen::Vector3f pos = T_wb.block<3, 1>(0, 3);
        
        glColor3f(1.0f, 0.0f, 0.0f);  // Red for current frame
        glLineWidth(3.0f);  // Thicker wireframe
        
        // Draw wireframe sphere for current frame
        glPushMatrix();
        glTranslatef(pos.x(), pos.y(), pos.z());
        GLUquadric* quad = gluNewQuadric();
        gluQuadricDrawStyle(quad, GLU_LINE);  // Wireframe style
        gluSphere(quad, 0.2, 16, 12);  // radius=0.2 (10x larger), slices=16, stacks=12
        gluDeleteQuadric(quad);
        glPopMatrix();
        
        glLineWidth(1.0f);  // Reset line width
    }
}

void VizUtils::DrawTrajectory(const std::vector<std::shared_ptr<Frame>>& frames) {
    if (frames.size() < 2) return;
    
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.0f, 0.0f);  // Red trajectory
    glBegin(GL_LINE_STRIP);
    
    for (const auto& frame : frames) {
        Eigen::Matrix4f T_wb = frame->GetTwb();
        Eigen::Vector3f pos = T_wb.block<3, 1>(0, 3);
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    
    glEnd();
}

void VizUtils::DrawKeyframes(const std::vector<std::shared_ptr<Frame>>& keyframes) {
    for (const auto& kf : keyframes) {
        Eigen::Matrix4f T_wb = kf->GetTwb();
        DrawCamera(T_wb, 0.0f, 0.0f, 1.0f, 0.05f);  // Blue for keyframes
    }
}

void VizUtils::DrawCamera(const Eigen::Matrix4f& T_wb, float r, float g, float b, float size) {
    glPushMatrix();
    
    // Apply transformation (World to Body)
    Eigen::Matrix4f T_bw = T_wb.inverse();
    glMultMatrixf(T_bw.data());
    
    glLineWidth(1.5f);
    glColor3f(r, g, b);
    
    // Draw wireframe sphere to represent 360 camera
    const float radius = size;
    const int slices = 16;  // Longitudinal divisions
    const int stacks = 8;   // Latitudinal divisions
    
    // Draw latitude circles
    for (int i = 0; i <= stacks; ++i) {
        float lat = M_PI * (-0.5f + (float)i / stacks);
        float z = radius * std::sin(lat);
        float r_lat = radius * std::cos(lat);
        
        glBegin(GL_LINE_LOOP);
        for (int j = 0; j <= slices; ++j) {
            float lon = 2.0f * M_PI * (float)j / slices;
            float x = r_lat * std::cos(lon);
            float y = r_lat * std::sin(lon);
            glVertex3f(x, y, z);
        }
        glEnd();
    }
    
    // Draw longitude circles
    for (int j = 0; j < slices; ++j) {
        float lon = 2.0f * M_PI * (float)j / slices;
        
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= stacks; ++i) {
            float lat = M_PI * (-0.5f + (float)i / stacks);
            float z = radius * std::sin(lat);
            float r_lat = radius * std::cos(lat);
            float x = r_lat * std::cos(lon);
            float y = r_lat * std::sin(lon);
            glVertex3f(x, y, z);
        }
        glEnd();
    }
    
    // Draw a small axis at camera center to show orientation
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.2f, 0.2f);
    glVertex3f(0, 0, 0);
    glVertex3f(radius * 0.5f, 0, 0);
    
    // Y axis - Green
    glColor3f(0.2f, 1.0f, 0.2f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, radius * 0.5f, 0);
    
    // Z axis - Blue (forward direction)
    glColor3f(0.2f, 0.2f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, radius * 0.5f);
    
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
    
    // Check if system is initialized (has any MapPoints)
    bool has_mappoints = false;
    for (size_t i = 0; i < features.size(); ++i) {
        auto mp = current_frame->GetMapPoint(static_cast<int>(i));
        if (mp && !mp->IsBad()) {
            has_mappoints = true;
            break;
        }
    }
    
    // Draw tracking lines
    if (previous_frame) {
        const auto& prev_features = previous_frame->GetFeatures();
        
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            
            // After initialization: only draw inliers with valid MapPoints
            if (has_mappoints) {
                auto mp = current_frame->GetMapPoint(static_cast<int>(i));
                if (!mp || mp->IsBad()) {
                    continue;  // Skip outliers
                }
            }
            
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
    int inlier_count = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        
        // After initialization: only draw inliers with valid MapPoints
        if (has_mappoints) {
            auto mp = current_frame->GetMapPoint(static_cast<int>(i));
            if (!mp || mp->IsBad()) {
                continue;  // Skip outliers
            }
        }
        
        cv::Point2f pt = feature->GetPixelCoord();
        cv::Scalar color = GetColorFromAge(feature->GetAge(), 10);
        cv::circle(vis_image, pt, 3, color, -1, cv::LINE_AA);
        inlier_count++;
    }
    
    // Add text information
    cv::putText(vis_image, "360 VIO Tracking", cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    std::string info = "Inliers: " + std::to_string(inlier_count) + " / " + std::to_string(features.size());
    cv::putText(vis_image, info, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    if (current_frame->IsKeyframe()) {
        cv::putText(vis_image, "KEYFRAME", cv::Point(10, 90),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }
    
    return vis_image;
}

} // namespace vio_360
