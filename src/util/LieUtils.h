/**
 * @file      LieUtils.h
 * @brief     Lie group utilities for SO(3) and SE(3) operations without Sophus dependency
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIE_UTILS_H
#define LIE_UTILS_H

#include <Eigen/Dense>
#include <cmath>

namespace vio_360 {

// Constants
constexpr float kEpsilon = 1e-6f;
constexpr float kPi = 3.14159f;
constexpr double kEpsilonD = 1e-10;
constexpr double kPiD = 3.14159265358979323846;

/**
 * @brief SO(3) Lie group utilities (float version)
 * 
 * Implements essential operations for 3D rotations:
 * - exp: axis-angle → rotation matrix
 * - log: rotation matrix → axis-angle  
 * - hat: vector → skew-symmetric matrix
 * - vee: skew-symmetric matrix → vector
 */
class SO3 {
public:
    SO3() = default;
    
    /// Create from rotation matrix (automatically normalizes to SO(3))
    explicit SO3(const Eigen::Matrix3f& R);
    
    /// Normalize current rotation matrix to ensure it's in SO(3)
    void Normalize();
    
    /// Create from axis-angle vector
    static SO3 Exp(const Eigen::Vector3f& omega);
    
    /// Convert to axis-angle vector
    Eigen::Vector3f Log() const;
    
    /// Get rotation matrix
    const Eigen::Matrix3f& Matrix() const { return m_matrix; }
    Eigen::Matrix3f& Matrix() { return m_matrix; }
    
    /// Composition
    SO3 operator*(const SO3& other) const {
        return SO3(m_matrix * other.m_matrix);
    }
    
    /// Apply rotation to vector
    Eigen::Vector3f operator*(const Eigen::Vector3f& v) const {
        return m_matrix * v;
    }
    
    /// Inverse
    SO3 Inverse() const {
        return SO3(m_matrix.transpose());
    }
    
    /// Identity
    static SO3 Identity() {
        return SO3(Eigen::Matrix3f::Identity());
    }
    
private:
    Eigen::Matrix3f m_matrix = Eigen::Matrix3f::Identity();
};

/**
 * @brief SE(3) Lie group utilities
 * 
 * Implements essential operations for 3D poses:
 * - exp: 6D twist → transformation matrix
 * - log: transformation matrix → 6D twist
 * - composition and inverse operations
 */
class SE3 {
public:
    SE3() = default;
    SE3(const SO3& rotation, const Eigen::Vector3f& translation) 
        : m_rotation(rotation), m_translation(translation) {}
    SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
        : m_rotation(R), m_translation(t) {}
    explicit SE3(const Eigen::Matrix4f& matrix);
    
    /// Create from transformation matrix
    static SE3 FromMatrix(const Eigen::Matrix4f& T);
    
    /// Create from 6D twist vector [rho, phi] where rho=translation, phi=rotation
    static SE3 Exp(const Eigen::Matrix<float, 6, 1>& xi);
    
    /// Convert to 6D twist vector
    Eigen::Matrix<float, 6, 1> Log() const;
    
    /// Get transformation matrix
    Eigen::Matrix4f Matrix() const;
    
    /// Get rotation part
    const SO3& Rotation() const { return m_rotation; }
    SO3& Rotation() { return m_rotation; }
    
    /// Get rotation matrix
    Eigen::Matrix3f RotationMatrix() const { return m_rotation.Matrix(); }
    
    /// Get translation part
    const Eigen::Vector3f& Translation() const { return m_translation; }
    Eigen::Vector3f& Translation() { return m_translation; }
    
    /// Composition
    SE3 operator*(const SE3& other) const {
        return SE3(m_rotation * other.m_rotation,
                   m_translation + m_rotation * other.m_translation);
    }
    
    /// Apply transformation to point
    Eigen::Vector3f operator*(const Eigen::Vector3f& p) const {
        return m_rotation * p + m_translation;
    }
    
    /// Inverse
    SE3 Inverse() const {
        SO3 R_inv = m_rotation.Inverse();
        return SE3(R_inv, R_inv * (-m_translation));
    }
    
    /// Identity
    static SE3 Identity() {
        return SE3(SO3::Identity(), Eigen::Vector3f::Zero());
    }
    
private:
    SO3 m_rotation;
    Eigen::Vector3f m_translation = Eigen::Vector3f::Zero();
};

// ===== Utility Functions =====

/**
 * @brief Convert vector to skew-symmetric matrix
 * @param v 3D vector
 * @return 3x3 skew-symmetric matrix
 */
Eigen::Matrix3f Hat(const Eigen::Vector3f& v);

/**
 * @brief Convert skew-symmetric matrix to vector
 * @param S 3x3 skew-symmetric matrix
 * @return 3D vector
 */
Eigen::Vector3f Vee(const Eigen::Matrix3f& S);

// ===== Double Precision Versions =====

/**
 * @brief SO(3) Lie group utilities (double version)
 */
class SO3d {
public:
    SO3d() = default;
    
    /// Create from rotation matrix (automatically normalizes to SO(3))
    explicit SO3d(const Eigen::Matrix3d& R);
    
    /// Normalize current rotation matrix to ensure it's in SO(3)
    void Normalize();
    
    /// Create from axis-angle vector
    static SO3d Exp(const Eigen::Vector3d& omega);
    
    /// Sophus-compatible lowercase version
    static SO3d exp(const Eigen::Vector3d& omega) { return Exp(omega); }
    
    /// Convert to axis-angle vector
    Eigen::Vector3d Log() const;
    
    /// Sophus-compatible lowercase version
    Eigen::Vector3d log() const { return Log(); }
    
    /// Skew-symmetric matrix (hat operator)
    static Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S <<     0.0, -v(2),  v(1),
              v(2),     0.0, -v(0),
             -v(1),  v(0),     0.0;
        return S;
    }
    
    /// Get rotation matrix
    const Eigen::Matrix3d& Matrix() const { return m_matrix; }
    Eigen::Matrix3d& Matrix() { return m_matrix; }
    
    /// Sophus-compatible lowercase matrix accessor
    Eigen::Matrix3d matrix() const { return m_matrix; }
    
    /// Composition
    SO3d operator*(const SO3d& other) const {
        return SO3d(m_matrix * other.m_matrix);
    }
    
    /// Apply rotation to vector
    Eigen::Vector3d operator*(const Eigen::Vector3d& v) const {
        return m_matrix * v;
    }
    
    /// Inverse
    SO3d Inverse() const {
        return SO3d(m_matrix.transpose());
    }
    
    /// Identity
    static SO3d Identity() {
        return SO3d(Eigen::Matrix3d::Identity());
    }
    
private:
    Eigen::Matrix3d m_matrix = Eigen::Matrix3d::Identity();
};

/**
 * @brief SE(3) Lie group utilities (double version)
 * 
 * Compatible with Sophus::SE3d interface for drop-in replacement
 */
class SE3d {
public:
    SE3d() = default;
    SE3d(const SO3d& rotation, const Eigen::Vector3d& translation) 
        : m_rotation(rotation), m_translation(translation) {}
    SE3d(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : m_rotation(R), m_translation(t) {}
    explicit SE3d(const Eigen::Matrix4d& matrix);
    
    /// Create from transformation matrix
    static SE3d FromMatrix(const Eigen::Matrix4d& T);
    
    /// Create from 6D twist vector [translation(3), rotation(3)] (Sophus/Ceres order)
    static SE3d exp(const Eigen::Matrix<double, 6, 1>& xi);
    
    /// Convert to 6D twist vector [translation(3), rotation(3)] (Sophus/Ceres order)
    Eigen::Matrix<double, 6, 1> log() const;
    
    /// Get transformation matrix
    Eigen::Matrix4d matrix() const;
    
    /// Get rotation part (SO3d object)
    const SO3d& so3() const { return m_rotation; }
    SO3d& so3() { return m_rotation; }
    
    /// Get rotation matrix
    Eigen::Matrix3d rotationMatrix() const { return m_rotation.Matrix(); }
    
    /// Get translation part
    const Eigen::Vector3d& translation() const { return m_translation; }
    Eigen::Vector3d& translation() { return m_translation; }
    
    /// Composition
    SE3d operator*(const SE3d& other) const {
        return SE3d(m_rotation * other.m_rotation,
                   m_translation + m_rotation * other.m_translation);
    }
    
    /// Apply transformation to point
    Eigen::Vector3d operator*(const Eigen::Vector3d& p) const {
        return m_rotation * p + m_translation;
    }
    
    /// Inverse
    SE3d inverse() const {
        SO3d R_inv = m_rotation.Inverse();
        return SE3d(R_inv, R_inv * (-m_translation));
    }
    
    /// Identity
    static SE3d Identity() {
        return SE3d(SO3d::Identity(), Eigen::Vector3d::Zero());
    }
    
private:
    SO3d m_rotation;
    Eigen::Vector3d m_translation = Eigen::Vector3d::Zero();
};

/**
 * @brief Convert vector to skew-symmetric matrix (double version)
 */
Eigen::Matrix3d Hatd(const Eigen::Vector3d& v);

/**
 * @brief Convert skew-symmetric matrix to vector (double version)
 */
Eigen::Vector3d Veed(const Eigen::Matrix3d& S);

// ===== Interpolation Functions =====

/**
 * @brief Spherical Linear Interpolation (SLERP) for rotations
 * @param R1 Start rotation matrix
 * @param R2 End rotation matrix
 * @param t Interpolation parameter [0, 1]
 * @return Interpolated rotation matrix
 */
inline Eigen::Matrix3f Slerp(const Eigen::Matrix3f& R1, const Eigen::Matrix3f& R2, float t) {
    // R_interp = R1 * exp(t * log(R1^T * R2))
    Eigen::Matrix3f R_rel = R1.transpose() * R2;
    SO3 so3_rel(R_rel);
    Eigen::Vector3f omega = so3_rel.Log();
    SO3 so3_interp = SO3::Exp(t * omega);
    return R1 * so3_interp.Matrix();
}

/**
 * @brief Linear interpolation for translations
 * @param t1 Start translation
 * @param t2 End translation
 * @param alpha Interpolation parameter [0, 1]
 * @return Interpolated translation
 */
inline Eigen::Vector3f Lerp(const Eigen::Vector3f& t1, const Eigen::Vector3f& t2, float alpha) {
    return (1.0f - alpha) * t1 + alpha * t2;
}

/**
 * @brief Interpolate SE(3) pose
 * @param T1 Start pose (4x4 transformation matrix)
 * @param T2 End pose (4x4 transformation matrix)
 * @param t Interpolation parameter [0, 1]
 * @return Interpolated pose
 */
inline Eigen::Matrix4f InterpolatePose(const Eigen::Matrix4f& T1, const Eigen::Matrix4f& T2, float t) {
    Eigen::Matrix3f R1 = T1.block<3, 3>(0, 0);
    Eigen::Matrix3f R2 = T2.block<3, 3>(0, 0);
    Eigen::Vector3f p1 = T1.block<3, 1>(0, 3);
    Eigen::Vector3f p2 = T2.block<3, 1>(0, 3);
    
    Eigen::Matrix4f T_interp = Eigen::Matrix4f::Identity();
    T_interp.block<3, 3>(0, 0) = Slerp(R1, R2, t);
    T_interp.block<3, 1>(0, 3) = Lerp(p1, p2, t);
    return T_interp;
}

} // namespace vio_360

#endif // LIE_UTILS_H
