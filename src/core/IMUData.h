/**
 * @file      IMUData.h
 * @brief     IMU data structure for 360-degree VIO
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

namespace vio_360 {

/**
 * @brief IMU measurement data structure
 */
struct IMUData {
    double timestamp;       // Timestamp in seconds
    
    // Linear acceleration in m/s^2 (in IMU/Body frame)
    float ax, ay, az;
    
    // Angular velocity in rad/s (in IMU/Body frame)
    float gx, gy, gz;
    
    IMUData()
        : timestamp(0.0)
        , ax(0.0f), ay(0.0f), az(0.0f)
        , gx(0.0f), gy(0.0f), gz(0.0f) {}
    
    IMUData(double t, float accel_x, float accel_y, float accel_z,
            float gyro_x, float gyro_y, float gyro_z)
        : timestamp(t)
        , ax(accel_x), ay(accel_y), az(accel_z)
        , gx(gyro_x), gy(gyro_y), gz(gyro_z) {}
};

} // namespace vio_360
