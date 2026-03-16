#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <eigen-5.0.0/Eigen/Dense>
#include <cmath>
#include <queue>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <random>
#include "orbit.h"
#include "debris.h"
#include "satellite.h"

int main() {
    // satellite orbit
    double a, e, i, omega, Omega, theta;
    a = 750 + 6371; // km
    e = 0.063;
    i = 45;
    omega = 0;
    Omega = 0;
    theta = 0;

    Orbit o;
    Eigen::VectorXd OE(6);
    OE << a, e, i, omega, Omega, theta;
    o.set_OE(OE);
    o.set_mu(0);
    std::cout<< "Satellite initial Cartesian state: \n";
    o.print_cartesian();

    // Create Satellite Object
    Satellite sat;
    sat.set_orbit(o);

    // Test Case: Known Debris ID 921 from prior simulation which hit 3 distinct sensors
    Eigen::MatrixXd debris_samples(1, 6);
    debris_samples.row(0) << 6672.330956, -0.010873, -0.080671, -0.041101, 7.310223, 1.780524;
    double final_time = 10.0; // seconds

    std::cout << "\nStarting Detection Simulation for Known Test Case...\n";
    std::vector<DetectionResult> detection_results = sat.detection_sim(debris_samples, final_time);

    std::cout << "\n============================================\n";
    std::cout << "Simulation Complete. Detections found: " << detection_results.size() << " out of " << debris_samples.rows() << "\n";
    for (const auto& res : detection_results) {
        std::cout << "Debris ID: " << res.debris_id << " | First Detection Time: " << res.first_detection_time << " s\n";
        for (const auto& det : res.detections) {
            std::cout << "   - Sensor " << det.sensor_id << " hit at " << det.time << " s\n";
        }
    }
    std::cout << "============================================\n";

    return 0;
}
