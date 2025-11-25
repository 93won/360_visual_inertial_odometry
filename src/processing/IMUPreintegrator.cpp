/**
 * @file      IMUPreintegrator.cpp
 * @brief     Implements IMU data preintegration for VIO
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "processing/IMUPreintegrator.h"
#include "processing/Estimator.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vio_360 {

// ============================================================================
// IMUPreintegration Implementation
// ============================================================================

IMUPreintegration::IMUPreintegration() {
    Reset();
}

void IMUPreintegration::Reset() {
    delta_R = Eigen::Matrix3f::Identity();
    delta_V = Eigen::Vector3f::Zero();
    delta_P = Eigen::Vector3f::Zero();
    
    J_Rg = Eigen::Matrix3f::Zero();
    J_Vg = Eigen::Matrix3f::Zero();
    J_Va = Eigen::Matrix3f::Zero();
    J_Pg = Eigen::Matrix3f::Zero();
    J_Pa = Eigen::Matrix3f::Zero();
    
    covariance = Eigen::Matrix<float, 15, 15>::Zero();
    
    gyro_bias = Eigen::Vector3f::Zero();
    accel_bias = Eigen::Vector3f::Zero();
    
    dt_total = 0.0;
}

bool IMUPreintegration::IsValid() const {
    return dt_total > 0.0 && !delta_R.hasNaN() && !delta_V.hasNaN() && !delta_P.hasNaN();
}

// ============================================================================
// IMUPreintegrator Implementation
// ============================================================================

IMUPreintegrator::IMUPreintegrator()
    : m_gyro_bias(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gravity(0.0f, 0.0f, -9.81f)  // Default gravity pointing down
    , m_gyro_noise(1.0e-4f)          // Default noise parameters
    , m_accel_noise(1.0e-3f)
    , m_gyro_bias_noise(1.0e-6f)
    , m_accel_bias_noise(1.0e-5f)
    , m_initialized(false) {
}

void IMUPreintegrator::Reset() {
    m_gyro_bias = Eigen::Vector3f::Zero();
    m_accel_bias = Eigen::Vector3f::Zero();
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    m_initialized = false;
}

void IMUPreintegrator::EstimateInitialBias(
    const std::vector<IMUData>& imu_measurements,
    float gravity_magnitude) {
    
    if (imu_measurements.empty()) {
        std::cerr << "[IMU_PREINTEGRATOR] No IMU measurements for bias estimation" << std::endl;
        return;
    }
    
    std::cout << "[IMU_PREINTEGRATOR] Estimating initial bias from " 
              << imu_measurements.size() << " measurements" << std::endl;
    
    // Sum all measurements
    Eigen::Vector3f gyro_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_sum = Eigen::Vector3f::Zero();
    
    for (const auto& imu : imu_measurements) {
        gyro_sum += Eigen::Vector3f(imu.gx, imu.gy, imu.gz);
        accel_sum += Eigen::Vector3f(imu.ax, imu.ay, imu.az);
    }
    
    // Average as bias estimate
    m_gyro_bias = gyro_sum / static_cast<float>(imu_measurements.size());
    
    // Estimate gravity direction from accelerometer
    // In static conditions, accelerometer measures gravity + bias
    Eigen::Vector3f accel_mean = accel_sum / static_cast<float>(imu_measurements.size());
    
    // Gravity direction = average accelerometer reading direction
    Eigen::Vector3f gravity_direction = accel_mean.normalized();
    float measured_gravity_mag = accel_mean.norm();
    
    // Update gravity vector (pointing opposite to measured acceleration)
    m_gravity = -gravity_direction * gravity_magnitude;
    
    // Accelerometer bias = measured - expected gravity
    m_accel_bias = accel_mean + m_gravity;
    
    std::cout << "[IMU_PREINTEGRATOR] Initial bias estimated:" << std::endl;
    std::cout << "  - Gyro bias: [" << m_gyro_bias.transpose() << "] rad/s" << std::endl;
    std::cout << "  - Accel bias: [" << m_accel_bias.transpose() << "] m/s²" << std::endl;
    std::cout << "  - Measured gravity magnitude: " << measured_gravity_mag << " m/s²" << std::endl;
    std::cout << "  - Expected gravity magnitude: " << gravity_magnitude << " m/s²" << std::endl;
    std::cout << "  - Gravity direction: [" << gravity_direction.transpose() << "]" << std::endl;
    
    m_initialized = true;
}

void IMUPreintegrator::SetBias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& accel_bias) {
    m_gyro_bias = gyro_bias;
    m_accel_bias = accel_bias;
}

void IMUPreintegrator::SetNoiseParameters(
    float gyro_noise,
    float accel_noise,
    float gyro_bias_noise,
    float accel_bias_noise) {
    
    m_gyro_noise = gyro_noise;
    m_accel_noise = accel_noise;
    m_gyro_bias_noise = gyro_bias_noise;
    m_accel_bias_noise = accel_bias_noise;
}

std::shared_ptr<IMUPreintegration> IMUPreintegrator::Preintegrate(
    const std::vector<IMUData>& imu_measurements,
    double start_time,
    double end_time) {
    
    if (imu_measurements.empty()) {
        std::cerr << "[IMU_PREINTEGRATOR] No IMU measurements provided" << std::endl;
        return nullptr;
    }
    
    // Create new preintegration object
    auto preint = std::make_shared<IMUPreintegration>();
    preint->gyro_bias = m_gyro_bias;
    preint->accel_bias = m_accel_bias;
    
    // Filter measurements within time range
    std::vector<IMUData> filtered_measurements;
    for (const auto& imu : imu_measurements) {
        if (imu.timestamp >= start_time && imu.timestamp < end_time) {
            filtered_measurements.push_back(imu);
        }
    }
    
    if (filtered_measurements.empty()) {
        std::cerr << "[IMU_PREINTEGRATOR] No IMU measurements in time range [" 
                  << start_time << ", " << end_time << ")" << std::endl;
        return nullptr;
    }
    
    // Integrate all measurements
    for (size_t i = 0; i < filtered_measurements.size(); ++i) {
        float dt;
        if (i == 0) {
            // Use average sampling rate for first sample
            dt = (filtered_measurements.size() > 1) ? 
                 static_cast<float>(filtered_measurements[1].timestamp - filtered_measurements[0].timestamp) : 0.002f;
        } else {
            dt = static_cast<float>(filtered_measurements[i].timestamp - filtered_measurements[i-1].timestamp);
        }
        
        // Clamp dt to reasonable range (0.5ms to 20ms)
        dt = std::max(0.0005f, std::min(dt, 0.02f));
        
        // Integrate measurement
        IntegrateMeasurement(preint, filtered_measurements[i], dt);
        
        // Update covariance
        UpdateCovariance(preint, dt);
        
        // Accumulate time
        preint->dt_total += dt;
    }
    
    return preint;
}

void IMUPreintegrator::IntegrateMeasurement(
    std::shared_ptr<IMUPreintegration> preint,
    const IMUData& imu,
    float dt) {
    
    // Bias-corrected measurements
    Eigen::Vector3f gyro(imu.gx, imu.gy, imu.gz);
    Eigen::Vector3f accel(imu.ax, imu.ay, imu.az);
    
    gyro -= preint->gyro_bias;
    accel -= preint->accel_bias;
    
    // Current state
    Eigen::Matrix3f R = preint->delta_R;
    Eigen::Vector3f V = preint->delta_V;
    Eigen::Vector3f P = preint->delta_P;
    
    // Update rotation using Rodrigues formula
    Eigen::Vector3f omega_dt = gyro * dt;
    Eigen::Matrix3f dR = Rodrigues(omega_dt);
    Eigen::Matrix3f Jr = RightJacobian(omega_dt);
    
    // Update Jacobians w.r.t gyro bias
    preint->J_Rg = -dR.transpose() * Jr * dt;
    preint->J_Vg = preint->J_Vg + preint->J_Va * SkewSymmetric(accel) * preint->J_Rg;
    preint->J_Pg = preint->J_Pg + preint->J_Pa * SkewSymmetric(accel) * preint->J_Rg + preint->J_Vg * dt;
    
    // Update rotation
    preint->delta_R = R * dR;
    
    // Update velocity
    Eigen::Vector3f dV = R * accel * dt;
    preint->delta_V = V + dV;
    preint->J_Va = preint->J_Va + R * dt;  // Jacobian w.r.t accel bias
    
    // Update position
    Eigen::Vector3f dP = V * dt + 0.5f * R * accel * dt * dt;
    preint->delta_P = P + dP;
    preint->J_Pa = preint->J_Pa + preint->J_Va * dt + 0.5f * R * dt * dt;  // Jacobian w.r.t accel bias
}

void IMUPreintegrator::UpdateCovariance(
    std::shared_ptr<IMUPreintegration> preint,
    float dt) {
    
    // Construct noise covariance matrices (6x6)
    Eigen::Matrix<float, 6, 6> Nga = Eigen::Matrix<float, 6, 6>::Zero();
    Nga.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * (m_gyro_noise * m_gyro_noise);
    Nga.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * (m_accel_noise * m_accel_noise);
    
    // Random walk noise (bias evolution)
    Eigen::Matrix<float, 6, 6> NgaWalk = Eigen::Matrix<float, 6, 6>::Zero();
    NgaWalk.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * (m_gyro_bias_noise * m_gyro_bias_noise * dt);
    NgaWalk.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * (m_accel_bias_noise * m_accel_bias_noise * dt);
    
    // State transition matrix A (9x9) and noise mapping B (9x6)
    Eigen::Matrix<float, 9, 9> A = Eigen::Matrix<float, 9, 9>::Identity();
    Eigen::Matrix<float, 9, 6> B = Eigen::Matrix<float, 9, 6>::Zero();
    
    // Get current rotation state
    Eigen::Matrix3f dR = preint->delta_R;
    
    // State coupling
    A.block<3,3>(6,3) = Eigen::Matrix3f::Identity() * dt;  // δP depends on δV
    
    // Noise mapping matrix B
    B.block<3,3>(3,3) = dR * dt;                            // Velocity noise from accel
    B.block<3,3>(6,3) = 0.5f * dR * dt * dt;                // Position noise from accel
    
    // Update covariance for preintegration states (9x9 block)
    preint->covariance.block<9,9>(0,0) = A * preint->covariance.block<9,9>(0,0) * A.transpose() 
                                        + B * Nga * B.transpose();
    
    // Update bias covariance (6x6 block) - simple random walk
    preint->covariance.block<6,6>(9,9) += NgaWalk;
}

void IMUPreintegrator::UpdatePreintegrationWithNewBias(
    std::shared_ptr<IMUPreintegration> preint,
    const Eigen::Vector3f& new_gyro_bias,
    const Eigen::Vector3f& new_accel_bias) {
    
    if (!preint || !preint->IsValid()) {
        return;
    }
    
    // Calculate bias change
    Eigen::Vector3f delta_bg = new_gyro_bias - preint->gyro_bias;
    Eigen::Vector3f delta_ba = new_accel_bias - preint->accel_bias;
    
    // Update using Jacobians (faster than re-integration)
    preint->delta_R = preint->delta_R * Rodrigues(preint->J_Rg * delta_bg);
    preint->delta_V = preint->delta_V + preint->J_Vg * delta_bg + preint->J_Va * delta_ba;
    preint->delta_P = preint->delta_P + preint->J_Pg * delta_bg + preint->J_Pa * delta_ba;
    
    // Update covariance: bias uncertainty should reduce after optimization
    float bias_reduction_factor = 0.9f;
    preint->covariance.block<3,3>(9, 9) *= bias_reduction_factor;   // Gyro bias
    preint->covariance.block<3,3>(12, 12) *= bias_reduction_factor; // Accel bias
    
    // Add regularization to prevent singular covariance
    float min_bias_variance = 1e-8f;
    for (int i = 9; i < 15; ++i) {
        preint->covariance(i, i) = std::max(preint->covariance(i, i), min_bias_variance);
    }
    
    // Update bias to new values
    preint->gyro_bias = new_gyro_bias;
    preint->accel_bias = new_accel_bias;
}

// ============================================================================
// Mathematical Utilities
// ============================================================================

Eigen::Matrix3f IMUPreintegrator::SkewSymmetric(const Eigen::Vector3f& v) const {
    Eigen::Matrix3f skew;
    skew <<     0.0f, -v.z(),  v.y(),
             v.z(),     0.0f, -v.x(),
            -v.y(),  v.x(),     0.0f;
    return skew;
}

Eigen::Matrix3f IMUPreintegrator::Rodrigues(const Eigen::Vector3f& omega) const {
    float theta = omega.norm();
    
    if (theta < 1e-6f) {
        // Small angle approximation
        return Eigen::Matrix3f::Identity() + SkewSymmetric(omega);
    }
    
    Eigen::Vector3f axis = omega / theta;
    Eigen::Matrix3f K = SkewSymmetric(axis);
    
    // Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    return Eigen::Matrix3f::Identity() 
         + std::sin(theta) * K 
         + (1.0f - std::cos(theta)) * K * K;
}

Eigen::Matrix3f IMUPreintegrator::RightJacobian(const Eigen::Vector3f& omega) const {
    float theta = omega.norm();
    
    if (theta < 1e-6f) {
        // Small angle approximation
        return Eigen::Matrix3f::Identity() - 0.5f * SkewSymmetric(omega);
    }
    
    Eigen::Vector3f axis = omega / theta;
    Eigen::Matrix3f K = SkewSymmetric(axis);
    
    // Right Jacobian: Jr = I - ((1-cos(θ))/θ)K + ((θ-sin(θ))/θ)K²
    return Eigen::Matrix3f::Identity() 
         - ((1.0f - std::cos(theta)) / theta) * K 
         + ((theta - std::sin(theta)) / theta) * K * K;
}

} // namespace vio_360
