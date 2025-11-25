/**
 * @file      IMUPreintegrator.h
 * @brief     Handles IMU data preintegration for Visual-Inertial Odometry
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @par Mathematical Foundation
 * Based on "IMU Preintegration on Manifold for Efficient Visual-Inertial 
 * Maximum-a-Posteriori Estimation" by Forster et al. (RSS 2015)
 */

#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vio_360 {

// Forward declaration
struct IMUData;

/**
 * @brief IMU preintegration measurement structure
 * 
 * Stores preintegrated measurements from frame i to frame j:
 * - delta_R: Rotation increment (body frame)
 * - delta_V: Velocity increment (body frame)
 * - delta_P: Position increment (body frame)
 * - Jacobians: For efficient bias updates
 * - Covariance: Measurement uncertainty
 */
struct IMUPreintegration {
    // Preintegrated measurements (from frame i to j)
    Eigen::Matrix3f delta_R;        ///< Rotation increment ΔR_ij
    Eigen::Vector3f delta_V;        ///< Velocity increment Δv_ij
    Eigen::Vector3f delta_P;        ///< Position increment Δp_ij
    
    // Jacobians w.r.t. bias (for efficient bias updates)
    Eigen::Matrix3f J_Rg;           ///< ∂ΔR/∂bg (gyro bias)
    Eigen::Matrix3f J_Vg, J_Va;     ///< ∂Δv/∂bg, ∂Δv/∂ba
    Eigen::Matrix3f J_Pg, J_Pa;     ///< ∂Δp/∂bg, ∂Δp/∂ba
    
    // Covariance matrix (15x15: ΔR, Δv, Δp, Δbg, Δba)
    Eigen::Matrix<float, 15, 15> covariance;
    
    // Bias used during preintegration
    Eigen::Vector3f gyro_bias;      ///< Gyroscope bias [rad/s]
    Eigen::Vector3f accel_bias;     ///< Accelerometer bias [m/s²]
    
    // Time span
    double dt_total;                ///< Total integration time [s]
    
    /// Constructor
    IMUPreintegration();
    
    /// Reset to initial state
    void Reset();
    
    /// Check if preintegration is valid
    bool IsValid() const;
};

/**
 * @brief Handles IMU data preintegration for VIO
 * 
 * Main responsibilities:
 * 1. Preintegrate IMU measurements between frames
 * 2. Manage IMU bias estimates (gyro and accel)
 * 3. Update preintegrations when bias changes
 * 4. Estimate initial bias from static measurements
 */
class IMUPreintegrator {
public:
    IMUPreintegrator();
    ~IMUPreintegrator() = default;
    
    // ============ Initialization ============
    
    /**
     * @brief Reset preintegrator to initial state
     */
    void Reset();
    
    /**
     * @brief Estimate initial IMU bias from static measurements
     * @param imu_measurements Static IMU measurements
     * @param gravity_magnitude Expected gravity magnitude (default: 9.81 m/s²)
     */
    void EstimateInitialBias(
        const std::vector<IMUData>& imu_measurements,
        float gravity_magnitude = 9.81f
    );
    
    // ============ Bias Management ============
    
    /**
     * @brief Set current IMU bias estimates
     * @param gyro_bias Gyroscope bias [rad/s]
     * @param accel_bias Accelerometer bias [m/s²]
     */
    void SetBias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& accel_bias);
    
    /**
     * @brief Get current gyroscope bias
     * @return Gyroscope bias [rad/s]
     */
    Eigen::Vector3f GetGyroBias() const { return m_gyro_bias; }
    
    /**
     * @brief Get current accelerometer bias
     * @return Accelerometer bias [m/s²]
     */
    Eigen::Vector3f GetAccelBias() const { return m_accel_bias; }
    
    // ============ Preintegration ============
    
    /**
     * @brief Preintegrate IMU measurements between two timestamps
     * 
     * Integrates angular velocity and linear acceleration measurements
     * to compute relative pose, velocity, and position changes.
     * 
     * @param imu_measurements IMU measurements to integrate
     * @param start_time Start timestamp [s]
     * @param end_time End timestamp [s]
     * @return Preintegrated IMU measurement (nullptr if failed)
     */
    std::shared_ptr<IMUPreintegration> Preintegrate(
        const std::vector<IMUData>& imu_measurements,
        double start_time,
        double end_time
    );
    
    /**
     * @brief Update preintegration when bias changes
     * 
     * Uses Jacobians to efficiently update preintegration without
     * re-integrating all measurements.
     * 
     * @param preint Preintegration to update
     * @param new_gyro_bias New gyroscope bias
     * @param new_accel_bias New accelerometer bias
     */
    void UpdatePreintegrationWithNewBias(
        std::shared_ptr<IMUPreintegration> preint,
        const Eigen::Vector3f& new_gyro_bias,
        const Eigen::Vector3f& new_accel_bias
    );
    
    // ============ Gravity Management ============
    
    /**
     * @brief Set gravity vector in world frame
     * @param gravity Gravity vector [m/s²]
     */
    void SetGravity(const Eigen::Vector3f& gravity) { m_gravity = gravity; }
    
    /**
     * @brief Get gravity vector in world frame
     * @return Gravity vector [m/s²]
     */
    Eigen::Vector3f GetGravity() const { return m_gravity; }
    
    // ============ Noise Parameters ============
    
    /**
     * @brief Set IMU noise parameters
     * @param gyro_noise Gyroscope noise density [rad/s/√Hz]
     * @param accel_noise Accelerometer noise density [m/s²/√Hz]
     * @param gyro_bias_noise Gyroscope bias random walk [rad/s²/√Hz]
     * @param accel_bias_noise Accelerometer bias random walk [m/s³/√Hz]
     */
    void SetNoiseParameters(
        float gyro_noise,
        float accel_noise,
        float gyro_bias_noise,
        float accel_bias_noise
    );
    
    // ============ Status ============
    
    /**
     * @brief Check if IMU has been initialized
     * @return True if initialized
     */
    bool IsInitialized() const { return m_initialized; }
    
    /**
     * @brief Set initialization status
     * @param initialized Initialization flag
     */
    void SetInitialized(bool initialized) { m_initialized = initialized; }

private:
    // ============ Integration Helpers ============
    
    /**
     * @brief Integrate single IMU measurement
     * @param preint Preintegration object to update
     * @param imu IMU measurement
     * @param dt Time step [s]
     */
    void IntegrateMeasurement(
        std::shared_ptr<IMUPreintegration> preint,
        const IMUData& imu,
        float dt
    );
    
    /**
     * @brief Update covariance during integration
     * @param preint Preintegration object
     * @param dt Time step [s]
     */
    void UpdateCovariance(
        std::shared_ptr<IMUPreintegration> preint,
        float dt
    );
    
    // ============ Mathematical Utilities ============
    
    /**
     * @brief Skew-symmetric matrix from vector
     * @param v Input vector
     * @return 3x3 skew-symmetric matrix [v]×
     */
    Eigen::Matrix3f SkewSymmetric(const Eigen::Vector3f& v) const;
    
    /**
     * @brief Rodrigues rotation formula
     * @param omega Rotation vector [rad]
     * @return Rotation matrix exp([ω]×)
     */
    Eigen::Matrix3f Rodrigues(const Eigen::Vector3f& omega) const;
    
    /**
     * @brief Right Jacobian for SO(3)
     * @param omega Rotation vector [rad]
     * @return Right Jacobian matrix Jr(ω)
     */
    Eigen::Matrix3f RightJacobian(const Eigen::Vector3f& omega) const;

private:
    // ============ State Variables ============
    
    /// Current gyroscope bias estimate [rad/s]
    Eigen::Vector3f m_gyro_bias;
    
    /// Current accelerometer bias estimate [m/s²]
    Eigen::Vector3f m_accel_bias;
    
    /// Gravity vector in world frame [m/s²]
    Eigen::Vector3f m_gravity;
    
    // ============ Noise Parameters ============
    
    float m_gyro_noise;         ///< Gyroscope noise density [rad/s/√Hz]
    float m_accel_noise;        ///< Accelerometer noise density [m/s²/√Hz]
    float m_gyro_bias_noise;    ///< Gyroscope bias random walk [rad/s²/√Hz]
    float m_accel_bias_noise;   ///< Accelerometer bias random walk [m/s³/√Hz]
    
    // ============ Status Flags ============
    
    bool m_initialized;         ///< Initialization flag
};

} // namespace vio_360
