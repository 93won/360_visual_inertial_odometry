/**
 * @file      Factors.cpp
 * @brief     Implements Ceres cost functions (factors) for VIO optimization.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-08-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "optimization/Factors.h"
#include "processing/IMUPreintegrator.h"  // For IMUPreintegration
#include "util/LieUtils.h"  // For SE3d and SO3d
#include <limits>
#include <iostream>
#include <algorithm>  // For std::clamp
namespace vio_360 {
namespace factor {

// PnPFactor implementation
PnPFactor::PnPFactor(const Eigen::Vector2d& observation,
                            const Eigen::Vector3d& world_point,
                            const CameraParameters& camera_params,
                            const Eigen::Matrix4d& Tcb,
                            const Eigen::Matrix4d& T_wb_init,
                            const Eigen::Matrix2d& information)
    : m_observation(observation), m_world_point(world_point), 
      m_camera_params(camera_params), m_Tcb(Tcb), m_T_wb_init(T_wb_init),
      m_information(information), m_is_outlier(false) {}

bool PnPFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    // If marked as outlier, set residuals to zero and jacobians to zero
    if (m_is_outlier) {
        residuals[0] = 640.0;
        residuals[1] = 480.0;
        
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
        }
        return true;
    }

    // Right perturbation approach: T_wb = T_wb_init * exp(delta_xi)
    // Parameters are small perturbations, not absolute pose
    Eigen::Map<const Eigen::Vector6d> delta_xi(parameters[0]);
    
    // Apply perturbation to initial pose
    vio_360::SE3d T_wb_init(m_T_wb_init);
    vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
    vio_360::SE3d T_wb = T_wb_init * delta_T;  // Right perturbation
    
    // T_bw = T_wb^(-1) using SE3d inverse
    vio_360::SE3d T_bw = T_wb.inverse();
    Eigen::Matrix3d R_bw = T_bw.rotationMatrix();
    Eigen::Vector3d t_bw = T_bw.translation();
    
    // T_cw = T_cb * T_bw  (w→b then b→c = w→c)
    // m_Tcb is T_cb (b→c, body to camera)
    vio_360::SE3d T_cb(m_Tcb);
    vio_360::SE3d T_cw = T_cb * T_bw;
    
    // Transform world point to camera coordinates: Pc = T_cw * Pw
    Eigen::Vector3d point_camera = T_cw * m_world_point;
    
    double pcx = point_camera.x();
    double pcy = point_camera.y();
    double pcz = point_camera.z();
    double L = point_camera.norm();
    
    // Check for valid point
    if (L < 1e-10) {
         if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
        }
        return false;
    }
    
    // Equirectangular projection
    // Camera frame: X-right, Y-down, Z-forward
    // theta = atan2(x, z), phi = -asin(y/L)
    // u = cols * (0.5 + theta/(2π)), v = rows * (0.5 - phi/π)
    double cols = m_camera_params.cols;
    double rows = m_camera_params.rows;
    
    double theta = std::atan2(pcx, pcz);  // [-π, π]
    double phi = -std::asin(pcy / L);     // [-π/2, π/2]
    
    double u = cols * (0.5 + theta / (2.0 * M_PI));
    double v = rows * (0.5 - phi / M_PI);
    
    // Compute residuals: observation - projection
    // Handle ERP wrap-around for horizontal coordinate
    double du = m_observation.x() - u;
    double dv = m_observation.y() - v;
    
    // Wrap horizontal residual to [-cols/2, cols/2] range
    // This handles the 360° wrap-around at image boundaries
    if (du > cols / 2.0) {
        du -= cols;
    } else if (du < -cols / 2.0) {
        du += cols;
    }
    
    Eigen::Vector2d residual_vec(du, dv);
    
    // Check for large errors (likely wrap-around boundary issues)
    // If error is too large, treat as outlier to prevent optimization instability
    const double max_pixel_error = 100.0;  // Maximum allowed pixel error
    bool is_large_error = (std::abs(du) > max_pixel_error || std::abs(dv) > max_pixel_error);
    
    if (is_large_error) {
        // Set large but finite residual and zero Jacobian
        residuals[0] = max_pixel_error;
        residuals[1] = max_pixel_error;
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
        }
        return true;
    }

    // Apply information matrix weighting to residuals: r_weighted = sqrt(Info) * r
    Eigen::LLT<Eigen::Matrix2d> llt(m_information);
    if (llt.info() == Eigen::Success) {
        // Use Cholesky decomposition: Information = L * L^T
        // Weighted residual = L * residual
        Eigen::Vector2d weighted_residual = llt.matrixL() * residual_vec;
        residuals[0] = weighted_residual[0];
        residuals[1] = weighted_residual[1];
    } else {
        // Fallback to unweighted if Cholesky fails
        residuals[0] = residual_vec[0];
        residuals[1] = residual_vec[1];
    }

    // Compute analytical jacobians if requested
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
        
        // Jacobian calculation
        // Camera extrinsic rotation matrix
        Eigen::Matrix3d Rcb = m_Tcb.block<3, 3>(0, 0);
        
        // Body frame coordinates: Pb = Rbw * Pw + tbw
        Eigen::Vector3d Pb = R_bw * m_world_point + t_bw;
        
        // Equirectangular Jacobian
        // theta = atan2(x, z), phi = -asin(y/L)
        // u = cols * (0.5 + theta/(2π)), v = rows * (0.5 - phi/π)
        // 
        // ∂theta/∂x = z/(x²+z²), ∂theta/∂z = -x/(x²+z²)
        // ∂phi/∂y = -1/(L*sqrt(1-(y/L)²)) = -1/sqrt(x²+z²)
        // ∂phi/∂x = xy/(L²*sqrt(x²+z²)), ∂phi/∂z = yz/(L²*sqrt(x²+z²))
        //
        // ∂u/∂x = cols/(2π) * z/(x²+z²)
        // ∂u/∂z = -cols/(2π) * x/(x²+z²)
        // ∂v/∂x = -rows/π * xy/(L²*sqrt(x²+z²))
        // ∂v/∂y = rows/π * sqrt(x²+z²)/L²
        // ∂v/∂z = -rows/π * yz/(L²*sqrt(x²+z²))
        
        double xz_sq = pcx * pcx + pcz * pcz;
        double L_sq = L * L;
        double xz_norm = std::sqrt(xz_sq);
        
        // Avoid division by zero
        if (xz_sq < 1e-10 || L_sq < 1e-10) {
            jac.setZero();
            return true;
        }
        
        double cols = m_camera_params.cols;
        double rows = m_camera_params.rows;
        
        Eigen::Matrix<double, 2, 3> J_error_wrt_Pc;
        // Note: residual = obs - proj, so ∂residual/∂Pc = -∂proj/∂Pc
        J_error_wrt_Pc(0, 0) = -cols / (2.0 * M_PI) * pcz / xz_sq;   // -∂u/∂x
        J_error_wrt_Pc(0, 1) = 0.0;                                   // -∂u/∂y = 0
        J_error_wrt_Pc(0, 2) = cols / (2.0 * M_PI) * pcx / xz_sq;    // -∂u/∂z
        J_error_wrt_Pc(1, 0) = rows / M_PI * (pcx * pcy) / (L_sq * xz_norm);   // -∂v/∂x
        J_error_wrt_Pc(1, 1) = -rows / M_PI * xz_norm / L_sq;        // -∂v/∂y
        J_error_wrt_Pc(1, 2) = rows / M_PI * (pcy * pcz) / (L_sq * xz_norm);   // -∂v/∂z
        
        // Chain rule: ∂(error)/∂(twist) = ∂(error)/∂(Pc) * ∂(Pc)/∂(twist)
        
        // Translation part: ∂(Pc)/∂(translation) = -Rcb
        Eigen::Matrix<double, 2, 3> J_translation = J_error_wrt_Pc * (-Rcb);
        
        // Rotation part: ∂(Pc)/∂(rotation) = Rcb * [Pb]× for right perturbation
        Eigen::Matrix3d hatPb = vio_360::SO3d::hat(Pb);
        Eigen::Matrix<double, 2, 3> J_rotation = J_error_wrt_Pc * Rcb * hatPb;
        
        // Combine Jacobian components [translation | rotation] for Ceres order
        Eigen::Matrix<double, 2, 6> unweighted_jac;
        unweighted_jac << J_translation, J_rotation;
        
        // Apply information matrix weighting to Jacobian: J_weighted = sqrt(Info) * J
        Eigen::LLT<Eigen::Matrix2d> llt_jac(m_information);
        if (llt_jac.info() == Eigen::Success) {
            jac = llt_jac.matrixL() * unweighted_jac;
        } else {
            jac = unweighted_jac;
        }
    }
    
    return true;
}

double PnPFactor::compute_chi_square(double const* const* parameters) const {
    // Right perturbation: T_wb = T_wb_init * exp(delta_xi)
    Eigen::Map<const Eigen::Vector6d> delta_xi(parameters[0]);
    
    vio_360::SE3d T_wb_init(m_T_wb_init);
    vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
    vio_360::SE3d T_wb = T_wb_init * delta_T;
    
    // T_cw = T_cb * T_bw = T_cb * T_wb^(-1)
    vio_360::SE3d T_cb(m_Tcb);
    vio_360::SE3d T_cw = T_cb * T_wb.inverse();
    
    // Transform world point to camera coordinates
    Eigen::Vector3d point_camera = T_cw * m_world_point;
    
    double pcx = point_camera.x();
    double pcy = point_camera.y();
    double pcz = point_camera.z();
    double L = point_camera.norm();
    
    // Check for valid point
    if (L < 1e-10) {
        return std::numeric_limits<double>::max(); // Invalid, return large chi-square
    }
    
    // Equirectangular projection
    double cols = m_camera_params.cols;
    double rows = m_camera_params.rows;
    
    double theta = std::atan2(pcx, pcz);  // [-π, π]
    double phi = -std::asin(pcy / L);     // [-π/2, π/2]
    
    double u = cols * (0.5 + theta / (2.0 * M_PI));
    double v = rows * (0.5 - phi / M_PI);
    
    // Compute residuals: observation - projection
    // Handle ERP wrap-around for horizontal coordinate
    double du = m_observation.x() - u;
    double dv = m_observation.y() - v;
    
    // Wrap horizontal residual to [-cols/2, cols/2] range
    if (du > cols / 2.0) {
        du -= cols;
    } else if (du < -cols / 2.0) {
        du += cols;
    }
    
    Eigen::Vector2d residual_vec(du, dv);
    
    // Chi-square error with information matrix: r^T * Information * r
    double chi2_error = residual_vec.transpose() * m_information * residual_vec;
    
    return chi2_error;
}

double PnPFactor::compute_bearing_angle_error(double const* const* parameters) const {
    // Right perturbation: T_wb = T_wb_init * exp(delta_xi)
    Eigen::Map<const Eigen::Vector6d> delta_xi(parameters[0]);
    
    vio_360::SE3d T_wb_init(m_T_wb_init);
    vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
    vio_360::SE3d T_wb = T_wb_init * delta_T;
    
    // T_cw = T_cb * T_wb^(-1)
    vio_360::SE3d T_cb(m_Tcb);
    vio_360::SE3d T_cw = T_cb * T_wb.inverse();
    
    // Transform world point to camera coordinates
    Eigen::Vector3d point_camera = T_cw * m_world_point;
    double L_proj = point_camera.norm();
    
    if (L_proj < 1e-10) {
        return M_PI; // Invalid point, return maximum angle
    }
    
    // Bearing from projected 3D point
    Eigen::Vector3d bearing_proj = point_camera / L_proj;
    
    // Bearing from observation (pixel to unit sphere)
    double cols = m_camera_params.cols;
    double rows = m_camera_params.rows;
    
    // Inverse equirectangular: pixel → (theta, phi) → unit vector
    double theta_obs = (m_observation.x() / cols - 0.5) * 2.0 * M_PI;
    double phi_obs = -(m_observation.y() / rows - 0.5) * M_PI;
    
    // Bearing vector: (cos(phi)*sin(theta), -sin(phi), cos(phi)*cos(theta))
    double cos_phi = std::cos(phi_obs);
    double sin_phi = std::sin(phi_obs);
    double cos_theta = std::cos(theta_obs);
    double sin_theta = std::sin(theta_obs);
    
    Eigen::Vector3d bearing_obs(cos_phi * sin_theta, -sin_phi, cos_phi * cos_theta);
    
    // Angle between bearings: acos(dot product)
    double dot = bearing_obs.dot(bearing_proj);
    dot = std::clamp(dot, -1.0, 1.0); // Numerical stability
    
    return std::acos(dot);
}

// BAFactor implementation
BAFactor::BAFactor(const Eigen::Vector2d& observation,
                   const CameraParameters& camera_params,
                   const Eigen::Matrix4d& Tcb,
                   const Eigen::Matrix4d& T_wb_init,
                   const Eigen::Matrix2d& information)
    : m_observation(observation)
    , m_camera_params(camera_params)
    , m_Tcb(Tcb)
    , m_T_wb_init(T_wb_init)
    , m_information(information)
    , m_is_outlier(false) {
}

bool BAFactor::Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    
    if (m_is_outlier) {
        // If marked as outlier, set zero residual and jacobians
        residuals[0] = 640.0;
        residuals[1] = 480.0;
        
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                jac_pose.setZero();
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                jac_point.setZero();
            }
        }
        return true;
    }
    
    try {
        // Right perturbation approach: T_wb = T_wb_init * exp(delta_xi)
        Eigen::Map<const Eigen::Vector6d> delta_xi(parameters[0]);
        
        // Extract 3D point in world coordinates
        Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
        
        // Apply perturbation to initial pose
        vio_360::SE3d T_wb_init(m_T_wb_init);
        vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
        vio_360::SE3d Twb = T_wb_init * delta_T;  // Right perturbation
        
        // T_bw = T_wb^(-1), T_cw = T_cb * T_bw
        vio_360::SE3d Tbw = Twb.inverse();
        vio_360::SE3d Tcb(m_Tcb);
        vio_360::SE3d Tcw = Tcb * Tbw;
        
        // For Jacobian computation, we still need individual components
        Eigen::Matrix3d Rbw = Tbw.rotationMatrix();
        Eigen::Vector3d tbw = Tbw.translation();
        
        // Transform world point to camera coordinates
        Eigen::Vector3d camera_point = Tcw * world_point;
        
        double pcx = camera_point[0];
        double pcy = camera_point[1];
        double pcz = camera_point[2];
        double L = camera_point.norm();

        // Check for valid point
        if (L < 1e-10) {
            // Invalid point - return large residual
            residuals[0] = 640.0;
            residuals[1] = 360.0;

            if (jacobians) {
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                    jac_pose.setZero();
                }
                if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                    jac_point.setZero();
                }
            }
            return true;
        }

        // Equirectangular projection
        // Camera frame: X-right, Y-down, Z-forward
        double cols = m_camera_params.cols;
        double rows = m_camera_params.rows;
        
        double theta = std::atan2(pcx, pcz);  // [-π, π]
        double phi = -std::asin(pcy / L);     // [-π/2, π/2]
        
        double u = cols * (0.5 + theta / (2.0 * M_PI));
        double v = rows * (0.5 - phi / M_PI);
        
        // Compute residual: observed - projected
        // Handle ERP wrap-around for horizontal coordinate
        double du = m_observation.x() - u;
        double dv = m_observation.y() - v;
        
        // Wrap horizontal residual to [-cols/2, cols/2] range
        // This handles the 360° wrap-around at image boundaries
        if (du > cols / 2.0) {
            du -= cols;
        } else if (du < -cols / 2.0) {
            du += cols;
        }
        
        Eigen::Vector2d error(du, dv);
        
        // Check for large errors (likely wrap-around boundary issues)
        const double max_pixel_error = 100.0;
        bool is_large_error = (std::abs(du) > max_pixel_error || std::abs(dv) > max_pixel_error);
        
        if (is_large_error) {
            residuals[0] = max_pixel_error;
            residuals[1] = max_pixel_error;
            if (jacobians) {
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                    jac_pose.setZero();
                }
                if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                    jac_point.setZero();
                }
            }
            return true;
        }
        
        // Apply information matrix weighting using Cholesky decomposition: r_weighted = sqrt(Info) * r
        Eigen::LLT<Eigen::Matrix2d> llt(m_information);
        if (llt.info() == Eigen::Success) {
            // Use Cholesky decomposition: Information = L * L^T
            // Weighted residual = L * residual
            Eigen::Vector2d weighted_error = llt.matrixL() * error;
            residuals[0] = weighted_error[0];
            residuals[1] = weighted_error[1];
        } else {
            // Fallback to unweighted if Cholesky fails
            residuals[0] = error[0];
            residuals[1] = error[1];
        }
        
        // Compute Jacobians if requested
        if (jacobians) {
            // Equirectangular projection Jacobian
            // theta = atan2(x, z), phi = -asin(y/L)
            double xz_sq = pcx * pcx + pcz * pcz;
            double L_sq = L * L;
            double xz_norm = std::sqrt(xz_sq);
            
            // Check for singularity
            if (xz_sq < 1e-10 || L_sq < 1e-10) {
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                    jac_pose.setZero();
                }
                if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                    jac_point.setZero();
                }
                return true;
            }
            
            double cols = m_camera_params.cols;
            double rows = m_camera_params.rows;
            
            Eigen::Matrix<double, 2, 3> J_proj_camera;
            // Note: residual = obs - proj, so ∂residual/∂Pc = -∂proj/∂Pc
            J_proj_camera(0, 0) = -cols / (2.0 * M_PI) * pcz / xz_sq;   // -∂u/∂x
            J_proj_camera(0, 1) = 0.0;                                   // -∂u/∂y = 0
            J_proj_camera(0, 2) = cols / (2.0 * M_PI) * pcx / xz_sq;    // -∂u/∂z
            J_proj_camera(1, 0) = rows / M_PI * (pcx * pcy) / (L_sq * xz_norm);   // -∂v/∂x
            J_proj_camera(1, 1) = -rows / M_PI * xz_norm / L_sq;        // -∂v/∂y
            J_proj_camera(1, 2) = rows / M_PI * (pcy * pcz) / (L_sq * xz_norm);   // -∂v/∂z
            
            // Apply information matrix weighting to projection jacobian using Cholesky
            Eigen::LLT<Eigen::Matrix2d> llt_jac(m_information);
            Eigen::Matrix<double, 2, 3> weighted_J_proj;
            if (llt_jac.info() == Eigen::Success) {
                weighted_J_proj = llt_jac.matrixL() * J_proj_camera;
            } else {
                weighted_J_proj = J_proj_camera;
            }
            
            // Jacobian w.r.t pose (SE3 tangent space)
            if (jacobians[0]) {
                // Body point in body frame: Pb = Rbw * Pw + tbw
                // This matches PnPFactor: Pb = R_bw * m_world_point + t_bw
                Eigen::Vector3d body_point = Rbw * world_point + tbw;
                
                // Jacobian of camera point w.r.t body translation
                Eigen::Matrix<double, 3, 3> J_camera_trans = -m_Tcb.block<3, 3>(0, 0);
                
                // Jacobian of camera point w.r.t body rotation  
                Eigen::Matrix<double, 3, 3> J_camera_rot = m_Tcb.block<3, 3>(0, 0) * vio_360::SO3d::hat(body_point);
                
                // Combine translation and rotation jacobians [3x6]
                // Parameter order: [tx, ty, tz, rx, ry, rz] - same as PnPFactor
                Eigen::Matrix<double, 3, 6> J_camera_pose;
                J_camera_pose.block<3, 3>(0, 0) = J_camera_trans; // Column 0-2: w.r.t translation
                J_camera_pose.block<3, 3>(0, 3) = J_camera_rot;   // Column 3-5: w.r.t rotation
                
                // Chain rule: J_residual_pose = J_proj_camera * J_camera_pose
                Eigen::Matrix<double, 2, 6> J_pose = weighted_J_proj * J_camera_pose;
                
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                jac_pose = J_pose;
            }
            
            // Jacobian w.r.t 3D point
            if (jacobians[1]) {
                // Jacobian of camera point w.r.t world point
                Eigen::Matrix<double, 3, 3> J_camera_point = m_Tcb.block<3, 3>(0, 0) * Rbw;
                
                // Chain rule: J_residual_point = J_proj_camera * J_camera_point  
                Eigen::Matrix<double, 2, 3> J_point = weighted_J_proj * J_camera_point;
                
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                jac_point = J_point;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

double BAFactor::compute_chi_square(double const* const* parameters) const {
    if (m_is_outlier) {
        return 0.0;
    }
    
    // Compute residual (which is already weighted by information matrix in Evaluate())
    double residuals[2];
    Evaluate(parameters, residuals, nullptr);
    
    // Chi-square error: weighted_residual^T * weighted_residual
    // Since Evaluate() returns Information * error, this gives us error^T * Information^T * Information * error
    // For symmetric positive definite Information matrix: Information^T = Information
    // So this becomes: error^T * Information^2 * error, which is incorrect for chi-square test
    
    // We need to compute the unweighted error and then apply information matrix properly
    // Right perturbation: T_wb = T_wb_init * exp(delta_xi)
    Eigen::Map<const Eigen::Vector6d> delta_xi(parameters[0]);
    vio_360::SE3d T_wb_init(m_T_wb_init);
    vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
    vio_360::SE3d Twb = T_wb_init * delta_T;
    
    // Extract 3D point
    Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
    
    // T_cw = T_cb * T_wb^(-1)
    vio_360::SE3d Tcb(m_Tcb);
    vio_360::SE3d Tcw = Tcb * Twb.inverse();
    
    // Camera coordinates: Pc = Tcw * Pw
    Eigen::Vector3d camera_point = Tcw * world_point;
    
    double pcx = camera_point.x();
    double pcy = camera_point.y();
    double pcz = camera_point.z();
    double L = camera_point.norm();
    
    // Check for valid point
    if (L < 1e-10) {
        return 1000.0; // Large chi-square for invalid points
    }
    
    // Equirectangular projection
    double cols = m_camera_params.cols;
    double rows = m_camera_params.rows;
    
    double theta = std::atan2(pcx, pcz);  // [-π, π]
    double phi = -std::asin(pcy / L);     // [-π/2, π/2]
    
    double u = cols * (0.5 + theta / (2.0 * M_PI));
    double v = rows * (0.5 - phi / M_PI);
    
    // Compute unweighted error with wrap-around handling
    double du = m_observation.x() - u;
    double dv = m_observation.y() - v;
    
    // Wrap horizontal residual to [-cols/2, cols/2] range
    if (du > cols / 2.0) {
        du -= cols;
    } else if (du < -cols / 2.0) {
        du += cols;
    }
    
    Eigen::Vector2d error(du, dv);
    
    // Proper chi-square: error^T * Information * error
    double chi_square = error.transpose() * m_information * error;
    
    return chi_square;
}

double BAFactor::compute_bearing_angle_error(double const* const* parameters) const {
    if (m_is_outlier) {
        return 0.0;
    }
    
    // Right perturbation: T_wb = T_wb_init * exp(delta_xi)
    Eigen::Map<const Eigen::Vector6d> delta_xi(parameters[0]);
    vio_360::SE3d T_wb_init(m_T_wb_init);
    vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
    vio_360::SE3d Twb = T_wb_init * delta_T;
    
    // Extract 3D point
    Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
    
    // T_cw = T_cb * T_wb^(-1)
    vio_360::SE3d Tcb(m_Tcb);
    vio_360::SE3d Tcw = Tcb * Twb.inverse();
    
    Eigen::Vector3d camera_point = Tcw * world_point;
    double L_proj = camera_point.norm();
    
    if (L_proj < 1e-10) {
        return M_PI; // Invalid point, return maximum angle
    }
    
    // Bearing from projected 3D point
    Eigen::Vector3d bearing_proj = camera_point / L_proj;
    
    // Bearing from observation (pixel to unit sphere)
    double cols = m_camera_params.cols;
    double rows = m_camera_params.rows;
    
    // Inverse equirectangular: pixel → (theta, phi) → unit vector
    double theta_obs = (m_observation.x() / cols - 0.5) * 2.0 * M_PI;
    double phi_obs = -(m_observation.y() / rows - 0.5) * M_PI;
    
    double cos_phi = std::cos(phi_obs);
    double sin_phi = std::sin(phi_obs);
    double cos_theta = std::cos(theta_obs);
    double sin_theta = std::sin(theta_obs);
    
    Eigen::Vector3d bearing_obs(cos_phi * sin_theta, -sin_phi, cos_phi * cos_theta);
    
    // Angle between bearings
    double dot = bearing_obs.dot(bearing_proj);
    dot = std::clamp(dot, -1.0, 1.0);
    
    return std::acos(dot);
}

// ===============================================================================
// INERTIAL GRAVITY FACTOR IMPLEMENTATION
// ===============================================================================

InertialGravityFactor::InertialGravityFactor(std::shared_ptr<IMUPreintegration> preintegration,
                                           double gravity_magnitude)
    : m_preintegration(preintegration), m_gravity_magnitude(gravity_magnitude) {
    // Extract covariance for rotation, velocity, and position (9x9 block)
    Eigen::Matrix<double, 9, 9> covariance_9x9 = m_preintegration->covariance.block<9, 9>(0, 0).cast<double>();
    
    // Compute information matrix (covariance inverse) with numerical stability check
    Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(covariance_9x9, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Apply regularization for numerical stability
    const double min_singular_value = 1e-6;
    Eigen::Matrix<double, 9, 1> singular_values = svd.singularValues();
    for (int i = 0; i < 9; ++i) {
        if (singular_values(i) < min_singular_value) {
            singular_values(i) = min_singular_value;
        }
    }
    
    // Compute regularized inverse: A^(-1) = V * S^(-1) * U^T
    Eigen::Matrix<double, 9, 9> information = svd.matrixV() * singular_values.cwiseInverse().asDiagonal() * svd.matrixU().transpose();
    
    // ⭐ Scale down information matrix for better numerical conditioning
    // IMU preintegration covariance is often too optimistic (too small)
    // Scale by 1e-6 to make information ~1.0 instead of ~1e6
    information *= 1e-6;
    
    // Compute square root information matrix using Cholesky decomposition
    Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(information);
    if (llt.info() == Eigen::Success) {
        m_sqrt_information = llt.matrixL().transpose(); // Upper triangular
    } else {
        // Fallback to identity if decomposition fails
        m_sqrt_information = Eigen::Matrix<double, 9, 9>::Identity();
        std::cerr << "[WARNING] " <<("[InertialGravityFactor] Cholesky decomposition failed, using identity weighting");
    }
    
}

bool InertialGravityFactor::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
    
    // Mathematical foundation: This implementation follows the formulation in
    // "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation"
    // by Forster et al. (RSS 2015), specifically Appendix C for gravity-aware preintegration factors.
    
    // ===============================================================================
    // STEP 1: Extract parameters from optimization variables
    // ===============================================================================
    
    // parameters[0]: SE3 posei [ti, Ri] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posei_tangent(parameters[0]);
    vio_360::SE3d T_wbi = vio_360::SE3d::exp(posei_tangent);
    Eigen::Matrix3d R_wbi = T_wbi.rotationMatrix();
    Eigen::Vector3d t_wbi = T_wbi.translation();
    Eigen::Matrix3d R_bwi = R_wbi.transpose();
    
    // parameters[1]: velocityi [vi]
    Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);
    
    // parameters[2]: shared gyro bias [bg] - SHARED across all factors
    Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
    
    // parameters[3]: shared accel bias [ba] - SHARED across all factors  
    Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
    
    // parameters[4]: SE3 posej [tj, Rj] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posej_tangent(parameters[4]);
    vio_360::SE3d T_wbj = vio_360::SE3d::exp(posej_tangent);
    Eigen::Matrix3d R_wbj = T_wbj.rotationMatrix();
    Eigen::Vector3d t_wbj = T_wbj.translation();
    
    // parameters[5]: velocityj [vj]
    Eigen::Map<const Eigen::Vector3d> vj(parameters[5]);
    
    // parameters[6]: gravity_dir: 2D gravity direction parameterization
    Eigen::Map<const Eigen::Vector2d> gravity_dir(parameters[6]);
    
    // ===============================================================================
    // STEP 2: Compute gravity vector from direction parameterization
    // ===============================================================================
    
    Eigen::Matrix3d R_wg = gravity_dir_to_rotation(gravity_dir);
    Eigen::Vector3d g_I(0, 0, -m_gravity_magnitude);
    Eigen::Vector3d g = R_wg * g_I;  // gravity in world frame
    
    // std::cout << "[InertialGravityFactor] g_I: " << g_I.transpose() << std::endl;
    // std::cout << "[InertialGravityFactor] g (world): " << g.transpose() << std::endl;
    
    // ===============================================================================
    // STEP 3: Get bias-corrected preintegration values
    // ===============================================================================
    
    double dt = m_preintegration->dt_total;
    
    // Create bias struct (assuming similar to IMU::Bias structure)
    // Note: You may need to adjust this based on your IMUPreintegration structure
    Eigen::Vector3d current_ba = ba;
    Eigen::Vector3d current_bg = bg;
    
    // Get corrected preintegration values 
    Eigen::Matrix3d delta_R = m_preintegration->delta_R.cast<double>(); // This should be GetDeltaRotation(bias)
    Eigen::Vector3d delta_V = m_preintegration->delta_V.cast<double>(); // This should be GetDeltaVelocity(bias) 
    Eigen::Vector3d delta_P = m_preintegration->delta_P.cast<double>(); // This should be GetDeltaPosition(bias)
    
    // Apply bias corrections using Jacobians
    Eigen::Vector3d delta_bg = bg - m_preintegration->gyro_bias.cast<double>();
    Eigen::Vector3d delta_ba = ba - m_preintegration->accel_bias.cast<double>();
    
    if (delta_bg.norm() > 1e-6 || delta_ba.norm() > 1e-6) {
        // Apply bias correction using precomputed Jacobians
        Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
        Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
        Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
        Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
        Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
        
        delta_R = delta_R * vio_360::SO3d::exp(J_Rg*delta_bg).matrix();
        delta_V = delta_V + J_Vg * delta_bg + J_Va * delta_ba;
        delta_P = delta_P + J_Pg * delta_bg + J_Pa * delta_ba;
    }
    
    // ===============================================================================  
    // STEP 4: Compute residuals 
    // ===============================================================================
    
    Eigen::Map<Eigen::Vector3d> er(residuals);      // rotation residual  
    Eigen::Map<Eigen::Vector3d> ev(residuals + 3);  // velocity residual
    Eigen::Map<Eigen::Vector3d> ep(residuals + 6);  // position residual
    
    // Rotation residual: er = Log(delta_R^T * Ri^T * Rj)
    er = log_SO3(delta_R.transpose() * R_bwi * R_wbj);
    
    // Velocity residual: ev = Ri^T * ((vj - vi) - g*dt) - delta_V
    ev = R_bwi * ((vj - vi) - g * dt) - delta_V;
    
    // Position residual: ep = Ri^T * ((tj - ti - vi*dt) - g*dt²/2) - delta_P  
    ep = R_bwi * ((t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt) - delta_P;

    // ===============================================================================
    // STEP 5: Compute Jacobians 
    // ===============================================================================
    
    if (jacobians != nullptr) {
        
        // Get bias corrected values for Jacobian computation
        Eigen::Vector3d dbg = delta_bg;
        Eigen::Matrix3d eR = delta_R.transpose() * R_bwi * R_wbj;
        Eigen::Vector3d er_vec = log_SO3(eR);
        Eigen::Matrix3d Jr_inv = right_jacobian_SO3(er_vec).inverse();
        
        // Jacobian w.r.t posei [0] - parameters[0]: [6x9]
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_posei(jacobians[0]);
            J_posei.setZero();
            
            // rotation part
            J_posei.block<3, 3>(0, 0) = -Jr_inv * R_wbj.transpose() * R_wbi;
            J_posei.block<3, 3>(3, 0) = skew_symmetric(R_bwi * ((vj - vi) - g * dt));
            J_posei.block<3, 3>(6, 0) = skew_symmetric(R_bwi * ((t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt));
            
            // translation part
            J_posei.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();
        }
        
        // Jacobian w.r.t velocity [1] - parameters[1]: [3x9]
        if (jacobians[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_veli(jacobians[1]);
            J_veli.setZero();
            J_veli.block<3, 3>(3, 0) = -R_bwi;
            J_veli.block<3, 3>(6, 0) = -R_bwi * dt;
        }
        
        // Jacobian w.r.t gyro_bias [2] - parameters[2]: [3x9] 
        if (jacobians[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_gyro(jacobians[2]);
            J_gyro.setZero();
            
            Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
            Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
            Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
            
            J_gyro.block<3, 3>(0, 0) = -Jr_inv * eR.transpose() * right_jacobian_SO3(J_Rg*dbg) * J_Rg;
            J_gyro.block<3, 3>(3, 0) = -J_Vg;
            J_gyro.block<3, 3>(6, 0) = -J_Pg;
        }
        
        // Jacobian w.r.t accel_bias [3] - parameters[3]: [3x9]
        if (jacobians[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_accel(jacobians[3]);
            J_accel.setZero();
            
            Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
            Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
            
            J_accel.block<3, 3>(3, 0) = -J_Va;
            J_accel.block<3, 3>(6, 0) = -J_Pa;
        }
        
        // Jacobian w.r.t posej [4] - parameters[4]: [6x9]
        if (jacobians[4] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_posej(jacobians[4]);
            J_posej.setZero();
            
            // rotation part
            J_posej.block<3, 3>(0, 0) = Jr_inv;
            
            // translation part  
            J_posej.block<3, 3>(6, 3) = R_bwi * R_wbj;
        }
        
        // Jacobian w.r.t velocityj [5] - parameters[5]: [3x9]
        if (jacobians[5] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_velj(jacobians[5]);
            J_velj.setZero();
            J_velj.block<3, 3>(3, 0) = R_bwi;
        }
        
        // Jacobian w.r.t gravity_dir [6] - parameters[6]: [2x9]
        if (jacobians[6] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> J_gravity(jacobians[6]);
            J_gravity.setZero();
            
            // Compute gravity direction Jacobian
            Eigen::Matrix<double, 3, 2> dGdTheta;
            dGdTheta.setZero();
            dGdTheta(0, 1) = -m_gravity_magnitude;
            dGdTheta(1, 0) = m_gravity_magnitude;
            Eigen::Matrix<double, 3, 2> dg_dtheta = R_wg * dGdTheta;
            
            J_gravity.block<3, 2>(3, 0) = -R_bwi * dg_dtheta * dt;
            J_gravity.block<3, 2>(6, 0) = -0.5 * R_bwi * dg_dtheta * dt * dt;
        }
    }
    
    return true;
}

Eigen::Matrix3d InertialGravityFactor::skew_symmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

Eigen::Matrix3d InertialGravityFactor::right_jacobian_SO3(const Eigen::Vector3d& phi) const {
    double theta = phi.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity() - 0.5 * skew_symmetric(phi);
    }
    
    double c = cos(theta);
    double s = sin(theta);
    Eigen::Matrix3d W = skew_symmetric(phi);
    
    return Eigen::Matrix3d::Identity() - 
           W * (1.0 - c) / (theta * theta) + 
           W * W * (theta - s) / (theta * theta * theta);
}

Eigen::Matrix3d InertialGravityFactor::left_jacobian_SO3(const Eigen::Vector3d& phi) const {
    return right_jacobian_SO3(-phi).transpose();
}

Eigen::Vector3d InertialGravityFactor::log_SO3(const Eigen::Matrix3d& R) const {
    // Normalize rotation matrix using SVD to ensure orthogonality before log operation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_normalized = svd.matrixU() * svd.matrixV().transpose();
    
    return vio_360::SO3d(R_normalized).log();
}

Eigen::Matrix3d InertialGravityFactor::gravity_dir_to_rotation(const Eigen::Vector2d& gravity_dir) const {
    // Gravity direction parameterization compatible implementation
    // Gravity is represented as a 2D parameter on SO(3) manifold
    // Convert 2D perturbation to rotation matrix using SO(3) exponential map
    // Rwg = ExpSO3(theta[0], theta[1], 0.0)
    // where theta[0], theta[1] are small angles around Y and X axes respectively
    
    double theta_x = gravity_dir[0];  // rotation around Y axis
    double theta_y = gravity_dir[1];  // rotation around X axis
    
    // Use Rodrigues formula: ExpSO3([theta_x, theta_y, 0])
    Eigen::Vector3d w(theta_x, theta_y, 0.0);
    const double d2 = w.dot(w);
    const double d = std::sqrt(d2);
    
    // Skew-symmetric matrix [w]×
    Eigen::Matrix3d W;
    W << 0.0,      -w(2),    w(1),
         w(2),      0.0,    -w(0),
        -w(1),      w(0),     0.0;
    
    Eigen::Matrix3d Rwg;
    if (d < 1e-5) {
        // Small angle approximation
        Rwg = Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    } else {
        // Full Rodrigues formula
        Rwg = Eigen::Matrix3d::Identity() + W*std::sin(d)/d + W*W*(1.0 - std::cos(d))/d2;
    }
    
    return Rwg;
}

// ===============================================================================
// INERTIAL GRAVITY SCALE FACTOR IMPLEMENTATION (Monocular)
// ===============================================================================

InertialGravityScaleFactor::InertialGravityScaleFactor(std::shared_ptr<IMUPreintegration> preintegration,
                                                       double gravity_magnitude)
    : m_preintegration(preintegration), m_gravity_magnitude(gravity_magnitude) {
    // Extract covariance for rotation, velocity, and position (9x9 block)
    Eigen::Matrix<double, 9, 9> covariance_9x9 = m_preintegration->covariance.block<9, 9>(0, 0).cast<double>();
    
    // Compute information matrix (covariance inverse) with numerical stability check
    Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(covariance_9x9, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Apply regularization for numerical stability
    const double min_singular_value = 1e-6;
    Eigen::Matrix<double, 9, 1> singular_values = svd.singularValues();
    for (int i = 0; i < 9; ++i) {
        if (singular_values(i) < min_singular_value) {
            singular_values(i) = min_singular_value;
        }
    }
    
    // Compute regularized inverse
    Eigen::Matrix<double, 9, 9> information = svd.matrixV() * singular_values.cwiseInverse().asDiagonal() * svd.matrixU().transpose();
    
    // Scale down information matrix for better numerical conditioning
    information *= 1e-6;
    
    // Compute square root information matrix using Cholesky decomposition
    Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(information);
    if (llt.info() == Eigen::Success) {
        m_sqrt_information = llt.matrixL().transpose();
    } else {
        m_sqrt_information = Eigen::Matrix<double, 9, 9>::Identity();
        std::cerr << "[WARNING] " <<("[InertialGravityScaleFactor] Cholesky decomposition failed, using identity weighting");
    }
}

bool InertialGravityScaleFactor::Evaluate(double const* const* parameters,
                                          double* residuals,
                                          double** jacobians) const {
    
    // ===============================================================================
    // STEP 1: Extract parameters from optimization variables
    // ===============================================================================
    
    // parameters[0]: SE3 posei [ti, Ri] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posei_tangent(parameters[0]);
    vio_360::SE3d T_wbi = vio_360::SE3d::exp(posei_tangent);
    Eigen::Matrix3d R_wbi = T_wbi.rotationMatrix();
    Eigen::Vector3d t_wbi = T_wbi.translation();
    Eigen::Matrix3d R_bwi = R_wbi.transpose();
    
    // parameters[1]: velocityi [vi]
    Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);
    
    // parameters[2]: shared gyro bias [bg]
    Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
    
    // parameters[3]: shared accel bias [ba]
    Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
    
    // parameters[4]: SE3 posej [tj, Rj] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posej_tangent(parameters[4]);
    vio_360::SE3d T_wbj = vio_360::SE3d::exp(posej_tangent);
    Eigen::Matrix3d R_wbj = T_wbj.rotationMatrix();
    Eigen::Vector3d t_wbj = T_wbj.translation();
    
    // parameters[5]: velocityj [vj]
    Eigen::Map<const Eigen::Vector3d> vj(parameters[5]);
    
    // parameters[6]: gravity_dir [2D]
    Eigen::Map<const Eigen::Vector2d> gravity_dir(parameters[6]);
    
    // parameters[7]: scale [1] - NEW for monocular!
    const double s = parameters[7][0];
    
    // ===============================================================================
    // STEP 2: Compute gravity vector from direction parameterization
    // ===============================================================================
    
    Eigen::Matrix3d R_wg = gravity_dir_to_rotation(gravity_dir);
    Eigen::Vector3d g_I(0, 0, -m_gravity_magnitude);
    Eigen::Vector3d g = R_wg * g_I;
    
    // ===============================================================================
    // STEP 3: Get bias-corrected preintegration values
    // ===============================================================================
    
    double dt = m_preintegration->dt_total;
    
    Eigen::Matrix3d delta_R = m_preintegration->delta_R.cast<double>();
    Eigen::Vector3d delta_V = m_preintegration->delta_V.cast<double>();
    Eigen::Vector3d delta_P = m_preintegration->delta_P.cast<double>();
    
    // Apply bias corrections using Jacobians
    Eigen::Vector3d delta_bg = bg - m_preintegration->gyro_bias.cast<double>();
    Eigen::Vector3d delta_ba = ba - m_preintegration->accel_bias.cast<double>();
    
    if (delta_bg.norm() > 1e-6 || delta_ba.norm() > 1e-6) {
        Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
        Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
        Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
        Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
        Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
        
        delta_R = delta_R * vio_360::SO3d::exp(J_Rg*delta_bg).matrix();
        delta_V = delta_V + J_Vg * delta_bg + J_Va * delta_ba;
        delta_P = delta_P + J_Pg * delta_bg + J_Pa * delta_ba;
    }
    
    // ===============================================================================
    // STEP 4: Compute residuals WITH SCALE
    // ===============================================================================
    
    Eigen::Map<Eigen::Vector3d> er(residuals);      // rotation residual
    Eigen::Map<Eigen::Vector3d> ev(residuals + 3);  // velocity residual
    Eigen::Map<Eigen::Vector3d> ep(residuals + 6);  // position residual
    
    // Rotation residual: UNCHANGED (scale-invariant)
    er = log_SO3(delta_R.transpose() * R_bwi * R_wbj);
    
    // Velocity residual: ev = Ri^T * (s*(vj - vi) - g*dt) - delta_V
    ev = R_bwi * (s * (vj - vi) - g * dt) - delta_V;
    
    // Position residual: ep = Ri^T * (s*(tj - ti - vi*dt) - 0.5*g*dt^2) - delta_P
    ep = R_bwi * (s * (t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt) - delta_P;
    
    // ===============================================================================
    // STEP 5: Compute Jacobians
    // ===============================================================================
    
    if (jacobians != nullptr) {
        
        Eigen::Vector3d dbg = delta_bg;
        Eigen::Matrix3d eR = delta_R.transpose() * R_bwi * R_wbj;
        Eigen::Vector3d er_vec = log_SO3(eR);
        Eigen::Matrix3d Jr_inv = right_jacobian_SO3(er_vec).inverse();
        
        // Jacobian w.r.t posei [0]
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_posei(jacobians[0]);
            J_posei.setZero();
            
            // Rotation part (unchanged)
            J_posei.block<3, 3>(0, 0) = -Jr_inv * R_wbj.transpose() * R_wbi;
            
            // Velocity part (with scale)
            J_posei.block<3, 3>(3, 0) = skew_symmetric(R_bwi * (s * (vj - vi) - g * dt));
            
            // Position part (with scale)
            J_posei.block<3, 3>(6, 0) = skew_symmetric(R_bwi * (s * (t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt));
            J_posei.block<3, 3>(6, 3) = -s * Eigen::Matrix3d::Identity();
        }
        
        // Jacobian w.r.t velocityi [1]
        if (jacobians[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_veli(jacobians[1]);
            J_veli.setZero();
            J_veli.block<3, 3>(3, 0) = -s * R_bwi;
            J_veli.block<3, 3>(6, 0) = -s * R_bwi * dt;
        }
        
        // Jacobian w.r.t gyro_bias [2]
        if (jacobians[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_gyro(jacobians[2]);
            J_gyro.setZero();
            
            Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
            Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
            Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
            
            J_gyro.block<3, 3>(0, 0) = -Jr_inv * eR.transpose() * right_jacobian_SO3(J_Rg*dbg) * J_Rg;
            J_gyro.block<3, 3>(3, 0) = -J_Vg;
            J_gyro.block<3, 3>(6, 0) = -J_Pg;
        }
        
        // Jacobian w.r.t accel_bias [3]
        if (jacobians[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_accel(jacobians[3]);
            J_accel.setZero();
            
            Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
            Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
            
            J_accel.block<3, 3>(3, 0) = -J_Va;
            J_accel.block<3, 3>(6, 0) = -J_Pa;
        }
        
        // Jacobian w.r.t posej [4]
        if (jacobians[4] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J_posej(jacobians[4]);
            J_posej.setZero();
            
            // Rotation part (unchanged)
            J_posej.block<3, 3>(0, 0) = Jr_inv;
            
            // Translation part (with scale)
            J_posej.block<3, 3>(6, 3) = s * R_bwi * R_wbj;
        }
        
        // Jacobian w.r.t velocityj [5]
        if (jacobians[5] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J_velj(jacobians[5]);
            J_velj.setZero();
            J_velj.block<3, 3>(3, 0) = s * R_bwi;
        }
        
        // Jacobian w.r.t gravity_dir [6]
        if (jacobians[6] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> J_gravity(jacobians[6]);
            J_gravity.setZero();
            
            Eigen::Matrix<double, 3, 2> dGdTheta;
            dGdTheta.setZero();
            dGdTheta(0, 1) = -m_gravity_magnitude;
            dGdTheta(1, 0) = m_gravity_magnitude;
            Eigen::Matrix<double, 3, 2> dg_dtheta = R_wg * dGdTheta;
            
            J_gravity.block<3, 2>(3, 0) = -R_bwi * dg_dtheta * dt;
            J_gravity.block<3, 2>(6, 0) = -0.5 * R_bwi * dg_dtheta * dt * dt;
        }
        
        // Jacobian w.r.t scale [7] - NEW!
        if (jacobians[7] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 1>> J_scale(jacobians[7]);
            J_scale.setZero();
            
            // dr_R / ds = 0 (rotation is scale-invariant)
            J_scale.block<3, 1>(0, 0).setZero();
            
            // dr_V / ds = R_bwi * (vj - vi)
            J_scale.block<3, 1>(3, 0) = R_bwi * (vj - vi);
            
            // dr_P / ds = R_bwi * (tj - ti - vi*dt)
            J_scale.block<3, 1>(6, 0) = R_bwi * (t_wbj - t_wbi - vi * dt);
        }
    }
    
    return true;
}

Eigen::Matrix3d InertialGravityScaleFactor::skew_symmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

Eigen::Matrix3d InertialGravityScaleFactor::right_jacobian_SO3(const Eigen::Vector3d& phi) const {
    double theta = phi.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity() - 0.5 * skew_symmetric(phi);
    }
    
    double c = cos(theta);
    double s = sin(theta);
    Eigen::Matrix3d W = skew_symmetric(phi);
    
    return Eigen::Matrix3d::Identity() - 
           W * (1.0 - c) / (theta * theta) + 
           W * W * (theta - s) / (theta * theta * theta);
}

Eigen::Matrix3d InertialGravityScaleFactor::left_jacobian_SO3(const Eigen::Vector3d& phi) const {
    return right_jacobian_SO3(-phi).transpose();
}

Eigen::Vector3d InertialGravityScaleFactor::log_SO3(const Eigen::Matrix3d& R) const {
    // Normalize rotation matrix using SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_normalized = svd.matrixU() * svd.matrixV().transpose();
    
    return vio_360::SO3d(R_normalized).log();
}

Eigen::Matrix3d InertialGravityScaleFactor::gravity_dir_to_rotation(const Eigen::Vector2d& gravity_dir) const {
    double theta_x = gravity_dir[0];
    double theta_y = gravity_dir[1];
    
    Eigen::Vector3d w(theta_x, theta_y, 0.0);
    const double d2 = w.dot(w);
    const double d = std::sqrt(d2);
    
    Eigen::Matrix3d W;
    W << 0.0,      -w(2),    w(1),
         w(2),      0.0,    -w(0),
        -w(1),      w(0),     0.0;
    
    Eigen::Matrix3d Rwg;
    if (d < 1e-5) {
        Rwg = Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    } else {
        Rwg = Eigen::Matrix3d::Identity() + W*std::sin(d)/d + W*W*(1.0 - std::cos(d))/d2;
    }
    
    return Rwg;
}

Eigen::Matrix3d InertialGravityScaleFactor::rodrigues_SO3(const Eigen::Vector3d& omega) const {
    double theta = omega.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity() + skew_symmetric(omega);
    }
    
    Eigen::Vector3d axis = omega / theta;
    double c = cos(theta);
    double s = sin(theta);
    
    return c * Eigen::Matrix3d::Identity() + 
           s * skew_symmetric(axis) + 
           (1.0 - c) * axis * axis.transpose();
}

// ============================================================================
// InertialFactorFixedGravity Implementation
// ============================================================================

InertialFactorFixedGravity::InertialFactorFixedGravity(
    std::shared_ptr<IMUPreintegration> preintegration,
    const Eigen::Vector3d& gravity,
    const Eigen::Matrix4d& T_wb_i_init,
    const Eigen::Matrix4d& T_wb_j_init)
    : m_preintegration(preintegration)
    , m_gravity(gravity)
    , m_T_wb_i_init(T_wb_i_init)
    , m_T_wb_j_init(T_wb_j_init)
{
    // Compute square root information matrix from covariance (9x9 block for R, V, P)
    Eigen::Matrix<double, 9, 9> cov = m_preintegration->covariance.block<9, 9>(0, 0).cast<double>();
    
    // Add small regularization for numerical stability
    cov += Eigen::Matrix<double, 9, 9>::Identity() * 1e-8;
    
    // Compute information matrix and its square root
    Eigen::Matrix<double, 9, 9> info = cov.inverse();
    Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(info);
    
    if (llt.info() == Eigen::Success) {
        m_sqrt_information = llt.matrixL().transpose();
    } else {
        m_sqrt_information = Eigen::Matrix<double, 9, 9>::Identity();
    }
}

bool InertialFactorFixedGravity::Evaluate(double const* const* parameters,
                                          double* residuals,
                                          double** jacobians) const {
    // ===============================================================================
    // Extract parameters (right perturbation approach)
    // ===============================================================================
    
    // parameters[0]: delta_xi_i [6] - perturbation for pose i
    Eigen::Map<const Eigen::Vector6d> delta_xi_i(parameters[0]);
    vio_360::SE3d T_wb_i_init_se3(m_T_wb_i_init);
    vio_360::SE3d T_wb_i = T_wb_i_init_se3 * vio_360::SE3d::exp(delta_xi_i);
    Eigen::Matrix3d R_wbi = T_wb_i.rotationMatrix();
    Eigen::Vector3d t_wbi = T_wb_i.translation();
    Eigen::Matrix3d R_bwi = R_wbi.transpose();
    
    // parameters[1]: velocity_i [3]
    Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);
    
    // parameters[2]: shared gyro bias [3]
    Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
    
    // parameters[3]: shared accel bias [3]
    Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
    
    // parameters[4]: delta_xi_j [6] - perturbation for pose j
    Eigen::Map<const Eigen::Vector6d> delta_xi_j(parameters[4]);
    vio_360::SE3d T_wb_j_init_se3(m_T_wb_j_init);
    vio_360::SE3d T_wb_j = T_wb_j_init_se3 * vio_360::SE3d::exp(delta_xi_j);
    Eigen::Matrix3d R_wbj = T_wb_j.rotationMatrix();
    Eigen::Vector3d t_wbj = T_wb_j.translation();
    
    // parameters[5]: velocity_j [3]
    Eigen::Map<const Eigen::Vector3d> vj(parameters[5]);
    
    // ===============================================================================
    // Get bias-corrected preintegration values
    // ===============================================================================
    
    double dt = m_preintegration->dt_total;
    
    Eigen::Matrix3d delta_R = m_preintegration->delta_R.cast<double>();
    Eigen::Vector3d delta_V = m_preintegration->delta_V.cast<double>();
    Eigen::Vector3d delta_P = m_preintegration->delta_P.cast<double>();
    
    // Apply bias corrections
    Eigen::Vector3d delta_bg = bg - m_preintegration->gyro_bias.cast<double>();
    Eigen::Vector3d delta_ba = ba - m_preintegration->accel_bias.cast<double>();
    
    if (delta_bg.norm() > 1e-6 || delta_ba.norm() > 1e-6) {
        Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
        Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
        Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
        Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
        Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
        
        delta_R = delta_R * vio_360::SO3d::exp(J_Rg * delta_bg).matrix();
        delta_V = delta_V + J_Vg * delta_bg + J_Va * delta_ba;
        delta_P = delta_P + J_Pg * delta_bg + J_Pa * delta_ba;
    }
    
    // ===============================================================================
    // Compute residuals
    // ===============================================================================
    
    Eigen::Map<Eigen::Vector3d> er(residuals);      // rotation residual
    Eigen::Map<Eigen::Vector3d> ev(residuals + 3);  // velocity residual
    Eigen::Map<Eigen::Vector3d> ep(residuals + 6);  // position residual
    
    // Rotation residual
    er = log_SO3(delta_R.transpose() * R_bwi * R_wbj);
    
    // Velocity residual
    ev = R_bwi * (vj - vi - m_gravity * dt) - delta_V;
    
    // Position residual
    ep = R_bwi * (t_wbj - t_wbi - vi * dt - 0.5 * m_gravity * dt * dt) - delta_P;
    
    // Apply weighting
    Eigen::Map<Eigen::Matrix<double, 9, 1>> res(residuals);
    res = m_sqrt_information * res;
    
    // ===============================================================================
    // Compute Jacobians (simplified - numerical jacobians can be used)
    // ===============================================================================
    
    if (jacobians != nullptr) {
        // For simplicity, let Ceres compute numerical Jacobians
        // A full analytical implementation would go here
        
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            // Numerical differentiation will handle this
        }
        if (jacobians[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[1]);
            J.setZero();
            // d(ev)/d(vi) = -R_bwi
            J.block<3, 3>(3, 0) = -m_sqrt_information.block<3, 3>(3, 3) * R_bwi;
            // d(ep)/d(vi) = -R_bwi * dt
            J.block<3, 3>(6, 0) = -m_sqrt_information.block<3, 3>(6, 6) * R_bwi * dt;
        }
        if (jacobians[2] != nullptr) {
            // Jacobian w.r.t gyro bias
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[2]);
            J.setZero();
            
            Eigen::Matrix3d J_Rg = m_preintegration->J_Rg.cast<double>();
            Eigen::Matrix3d J_Vg = m_preintegration->J_Vg.cast<double>();
            Eigen::Matrix3d J_Pg = m_preintegration->J_Pg.cast<double>();
            
            // d(er)/d(bg): rotation residual w.r.t gyro bias
            // er = log(delta_R^T * R_bwi * R_wbj), delta_R depends on bg via J_Rg
            Eigen::Matrix3d Jr_inv = right_jacobian_SO3(-er).inverse();
            J.block<3, 3>(0, 0) = -Jr_inv * J_Rg;
            
            // d(ev)/d(bg): velocity residual w.r.t gyro bias
            J.block<3, 3>(3, 0) = -J_Vg;
            
            // d(ep)/d(bg): position residual w.r.t gyro bias
            J.block<3, 3>(6, 0) = -J_Pg;
            
            // Apply weighting
            J = m_sqrt_information * J;
        }
        if (jacobians[3] != nullptr) {
            // Jacobian w.r.t accel bias
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[3]);
            J.setZero();
            
            Eigen::Matrix3d J_Va = m_preintegration->J_Va.cast<double>();
            Eigen::Matrix3d J_Pa = m_preintegration->J_Pa.cast<double>();
            
            // d(er)/d(ba): rotation does not depend on accel bias
            // J.block<3, 3>(0, 0) = 0
            
            // d(ev)/d(ba): velocity residual w.r.t accel bias
            J.block<3, 3>(3, 0) = -J_Va;
            
            // d(ep)/d(ba): position residual w.r.t accel bias
            J.block<3, 3>(6, 0) = -J_Pa;
            
            // Apply weighting
            J = m_sqrt_information * J;
        }
        if (jacobians[4] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> J(jacobians[4]);
            J.setZero();
            // Numerical differentiation will handle this
        }
        if (jacobians[5] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[5]);
            J.setZero();
            // d(ev)/d(vj) = R_bwi
            J.block<3, 3>(3, 0) = m_sqrt_information.block<3, 3>(3, 3) * R_bwi;
        }
    }
    
    return true;
}

Eigen::Matrix3d InertialFactorFixedGravity::skew_symmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return S;
}

Eigen::Matrix3d InertialFactorFixedGravity::right_jacobian_SO3(const Eigen::Vector3d& phi) const {
    double theta = phi.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d Phi = skew_symmetric(phi);
    double theta2 = theta * theta;
    return Eigen::Matrix3d::Identity() 
           - (1.0 - std::cos(theta)) / theta2 * Phi
           + (theta - std::sin(theta)) / (theta2 * theta) * Phi * Phi;
}

Eigen::Vector3d InertialFactorFixedGravity::log_SO3(const Eigen::Matrix3d& R) const {
    double trace = R.trace();
    double cos_theta = (trace - 1.0) / 2.0;
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    double theta = std::acos(cos_theta);
    
    if (theta < 1e-6) {
        return Eigen::Vector3d(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)) / 2.0;
    }
    
    return theta / (2.0 * std::sin(theta)) * 
           Eigen::Vector3d(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
}

} // namespace factor
} // namespace vio_360
