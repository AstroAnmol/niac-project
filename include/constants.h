#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants {
    // Math constants
    constexpr double PI = 3.14159265358979323846;

    // Gravitational parameters (km^3/s^2)
    constexpr double GM_EARTH = 398600.4418;  // Earth
    constexpr double GM_SUN = 132712440018.0; // Sun

    // Radius of central body (km)
    constexpr double R_EARTH = 6378.137;  // Earth

    // Oblateness coefficient
    constexpr double J2_EARTH = 1.08262668e-03;  // Earth

    // Eccentricity limits for LEO debris
    constexpr double E_MAX_LEO = 0.15;

    // Inclination limits (degrees)
    constexpr double I_MIN_LEO = 0.0;
    constexpr double I_MAX_LEO = 180.0;

    // Altitude limits for LEO debris (km)
    constexpr double H_MIN_LEO = 200.0;
    constexpr double H_MAX_LEO = 2000.0;

    // RAAN, Argument of Perigee, True Anomaly limits (radians)
    constexpr double ANGLE_MIN = 0.0;
    constexpr double ANGLE_MAX = 2.0 * PI;

    // Soliton parameters
    constexpr double SOL_SHELL_THICKNESS = 1e-06;  // 1 cm 

    // Detection frequency (Hz)
    constexpr double DETECTION_FREQ = 10000.0;  // 10 kHz
}

#endif // CONSTANTS_H
