#define _USE_MATH_DEFINES

#include "debris.h"
#include "monte_carlo.h"
#include "orbit.h"
#include "satellite.h"
#include <cmath>
#include <eigen-5.0.0/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

int main() {

  // Monte Carlo Simulation
  // MonteCarlo mc;
  // int num_trials = 1000;

  // // Example for Inclination (i) based on LEO clusters:
  // std::vector<std::tuple<double, double, double>> InclinationMixture = {
  //     // Sun-Synchronous Cluster
  //     {0.40, 98.0, 1.5},
  //     // High-Inclination Cluster
  //     {0.30, 82.0, 7.0},
  //     // Mid-Inclination Cluster
  //     {0.30, 50.0, 10.0}
  // };

  // mc.define_gmms("i", InclinationMixture);
  // mc.define_gmms("a", { {1.0, 7000.0, 100.0} });
  // mc.define_eccentricity_params(-6.5, 1.0);
  // mc.run_simulation(num_trials, "monte_carlo");

  // satellite orbit
  double a, e, i, omega, Omega, theta;
  a = 750 + 6371; // km;
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
  std::cout << "Satellite initial Cartesian state: \n";
  o.print_cartesian();

  // Create Satellite Object
  Satellite sat;
  sat.set_orbit(o);

  // set wake angle
  // sat.set_wake_angle(270.0); // degrees

  // Generate Debris Samples and Simulate Detections!
  int num_samples = 10000;
  double search_radius = 1.0; // km
  double final_time = 10.0;   // seconds

  std::cout << "\nStarting Detection Simulation for " << num_samples
            << " debris samples...\n";
  auto sim_output = sat.detection_sim(num_samples, search_radius, final_time);
  Eigen::MatrixXd debris_samples = sim_output.first;
  std::vector<DetectionResult> detection_results = sim_output.second;

  std::cout << "\n============================================\n";
  std::cout << "Simulation Complete. Detections found: "
            << detection_results.size() << " out of " << num_samples << "\n";
  std::cout << "============================================\n";

  // Save to files in Results/ with timestamp
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << "Results/debris_samples_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
      << ".csv";
  std::string filename_debris = oss.str();

  std::ofstream outfile(filename_debris);
  outfile << std::setprecision(15);
  outfile << "x,y,z,vx,vy,vz\n";

  // Save Satellite first
  Orbit sat_o = sat.get_orbit();
  Eigen::VectorXd sat_st = sat_o.get_cartesian();
  outfile << sat_st(0) << "," << sat_st(1) << "," << sat_st(2) << ","
          << sat_st(3) << "," << sat_st(4) << "," << sat_st(5) << "\n";

  // Save Debris Samples
  for (int i = 0; i < debris_samples.rows(); ++i) {
    outfile << debris_samples(i, 0) << "," << debris_samples(i, 1) << ","
            << debris_samples(i, 2) << "," << debris_samples(i, 3) << ","
            << debris_samples(i, 4) << "," << debris_samples(i, 5) << "\n";
  }
  outfile.close();
  std::cout << "Debris samples saved to " << filename_debris << "\n";

  std::ostringstream oss2;
  oss2 << "Results/detection_results_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
       << ".csv";
  std::string filename_results = oss2.str();

  std::ofstream resfile(filename_results);
  resfile << std::setprecision(15);
  resfile << "debris_id,detected,first_detection_time,sensor_hits\n";
  for (const auto &res : detection_results) {
    resfile << res.debris_id << "," << (res.detected ? "true" : "false") << ","
            << res.first_detection_time << ",";
    for (size_t det_idx = 0; det_idx < res.detections.size(); ++det_idx) {
      resfile << "S" << res.detections[det_idx].sensor_id << "@"
              << res.detections[det_idx].time;
      if (det_idx < res.detections.size() - 1) {
        resfile << ";";
      }
    }
    resfile << "\n";
  }
  resfile.close();
  std::cout << "Detection results saved to " << filename_results << "\n";

  // double time_period = sat.get_TimePeriod();
  // // propagate satellite orbit at higher time step for plot
  // double step_sat = 10; // seconds
  // // sat.propagate_2BP(step_sat, time_period, 0, "sat_orbit");

  // // soliton characteristics
  // double cone_angle = 45 * M_PI / 180; // radians
  // double cone_height = 10;              // km
  // double sol_vel_multiplier = 1.2;      // arbitrary multiplier
  // double detection_freq = 10000; // Hz
  // double time_step = 1/detection_freq; // seconds

  // // debris object
  // Eigen::Vector3d r1, v1;
  // double a_d, e_d, i_d, omega_d, Omega_d, theta_d;
  // a_d =         a; // km;
  // e_d =         e;
  // i_d =         180-i;
  // omega_d =     180 + omega;
  // Omega_d =     180 + Omega;
  // theta_d =     theta - 0.1;

  // Eigen::VectorXd OE_d(6);
  // OE_d << a_d, e_d, i_d, omega_d, Omega_d, theta_d;
  // Orbit debris;
  // debris.set_OE(OE_d);
  // debris.set_mu(0);
  // debris.print_cartesian();
  // Eigen::VectorXd cartesian_d = debris.get_cartesian();
  // r1 = cartesian_d.segment(0,3);
  // v1 = cartesian_d.segment(3,3);

  // std::cout << "----------------------------------------\n";
  // std::cout << "Satellite Object Detection Simulation\n";
  // std::cout << "----------------------------------------\n";

  // Satellite satellite;
  // satellite.set_soliton_state(r1, v1);
  // satellite.detect_soliton_over_time();

  // std::cout << "----------------------------------------\n";
  // std::cout << "Debris Object Detection Simulation\n";
  // std::cout << "----------------------------------------\n";

  // Debris D1;
  // D1.set_state(r1, v1);
  // D1.set_soliton_params(cone_angle, cone_height, sol_vel_multiplier);
  // D1.set_detection_freq(detection_freq);

  // std::cout << "Soliton velocity: " << D1.get_soliton_vel() << " km/s" <<
  // std::endl;
  // // get time to reach cone base
  // double time_to_cone_base = D1.get_time_to_reach_cone_base();
  // std::cout << "Time to reach cone base: " << time_to_cone_base << " seconds"
  // << std::endl;

  // // propagate satellite orbit to the same time
  // sat.propagate_2BP(time_step, time_to_cone_base, 0, "sat_prop");

  // D1.read_sat_orbit("sat_prop");

  // D1.check_detection();
}