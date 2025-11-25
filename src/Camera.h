/**
 * @file      Camera.h
 * @brief     Defines the Camera class for Equirectangular (360-degree) camera model.
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

namespace vio_360 {

/**
 * @brief Camera model for Equirectangular Projection (ERP) 360-degree cameras
 */
class Camera {
public:
    Camera(int width, int height);
    ~Camera() = default;

    // ============ Projection Functions ============
    
    /**
     * @brief Convert pixel coordinates to bearing vector on unit sphere
     * @param pixel Pixel coordinates (u, v)
     * @return 3D unit bearing vector
     */
    Eigen::Vector3f PixelToBearing(const cv::Point2f& pixel) const;
    
    /**
     * @brief Convert bearing vector to pixel coordinates
     * @param bearing 3D unit bearing vector
     * @return Pixel coordinates (u, v)
     */
    cv::Point2f BearingToPixel(const Eigen::Vector3f& bearing) const;
    
    /**
     * @brief Compute angular distance between two bearing vectors
     * @param bearing1 First bearing vector
     * @param bearing2 Second bearing vector
     * @return Angular distance in radians
     */
    float AngularDistance(const Eigen::Vector3f& bearing1, 
                         const Eigen::Vector3f& bearing2) const;
    
    // ============ Mask Generation ============
    
    /**
     * @brief Create mask to exclude polar regions
     * @param top_ratio Ratio of top region to exclude (default 0.15)
     * @param bottom_ratio Ratio of bottom region to exclude (default 0.15)
     * @return Mask image (255 = valid, 0 = invalid)
     */
    cv::Mat CreatePolarMask(float top_ratio = 0.15f, float bottom_ratio = 0.15f) const;
    
    /**
     * @brief Check if pixel is in polar region (distorted area)
     * @param pixel Pixel coordinates
     * @param threshold Threshold ratio (default 0.15)
     * @return True if in polar region
     */
    bool IsInPolarRegion(const cv::Point2f& pixel, float threshold = 0.15f) const;
    
    // ============ Boundary Handling ============
    
    /**
     * @brief Handle horizontal wrapping for ERP images
     * @param pixel Pixel coordinates (will be modified in-place)
     */
    void WrapHorizontal(cv::Point2f& pixel) const;
    
    /**
     * @brief Check if pixel is near horizontal boundary
     * @param pixel Pixel coordinates
     * @param margin Margin in pixels (default 50)
     * @return True if near boundary
     */
    bool IsNearBoundary(const cv::Point2f& pixel, float margin = 50.0f) const;
    
    // ============ Getters ============
    
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    
private:
    int m_width;   // Image width
    int m_height;  // Image height
};

} // namespace vio_360
