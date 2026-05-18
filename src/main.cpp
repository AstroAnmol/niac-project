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


  std::cout << "Starting simulation...\n";
  // Intitalize a output file
  std::ofstream output_file("Results/output.txt");
  if (!output_file.is_open()) {
    std::cerr << "Error: Could not open output file.\n";
    return 1;
  }
  output_file << "Starting simulation...\n";
  output_file << "Initializing all constant parameters.\n";
  // satellite orbit
  double a, e, i, omega, Omega;
  a = 750 + 6371; // km;
  e = 0.063;
  i = 135;
  omega = 0;
  Omega = 0;

  output_file << "Satellite orbital elements: a=" << a << ", e=" << e << ", i=" << i << ", omega=" << omega << ", Omega=" << Omega << "\n";

  // Soliton parameters: cone angle (radians), cone height (km), velocity multiplier
  Eigen::Vector3d soliton_params;
  soliton_params << 10.0*M_PI / 180.0, 10.0, 1.2;
  // soliton_params << 45.0*M_PI / 180.0, 0.5, 1.2;
  output_file << "Soliton parameters: angle=" << soliton_params(0) * 180.0 / M_PI << " degrees, height=" << soliton_params(1) << " km, velocity multiplier=" << soliton_params(2) << "\n";

  Eigen::VectorXd boom_angles(8);
  // boom_angles << 00, 90, 00, 90, 00, 90, 00, 90; // all booms at [1, 0, 0]
  boom_angles << 90, 90, 90, 90, 90, 90, 90, 90; // all booms at [0, 1, 0]
  // boom_angles << 90, 00, 90, 00, 90, 00, 90, 00; // all booms at [0, 0, 1] 
  // boom_angles << 54.7356, 45.0, 54.7356, 45.0, 54.7356, 45.0, 54.7356, 45.0; // booms at tetrahedral angles

  output_file << "Boom angles: " << boom_angles.transpose() << "\n";
  boom_angles = boom_angles* M_PI / 180.0; // Convert to radians

  // Debris parameters

  // Generate Debris Samples and Simulate Detections!
  int num_samples = 10000;
  double search_radius = 10.0; // km
  double final_time = 2.0;   // seconds

  output_file << "Debris parameters: num_samples=" << num_samples << ", search_radius=" << search_radius << " km, final_time=" << final_time << " s\n";

  output_file << "Starting detection simulation using J2 propagation...\n";
  std::cout << "Starting detection simulation using J2 propagation...\n";
  
  // Set up J2 propagation with odeint
  Orbit o_ref;
  Eigen::VectorXd OE_initial(6);
  OE_initial << a, e, i, omega, Omega, 0;  // Start at nu=0
  o_ref.set_OE(OE_initial);
  o_ref.set_mu(0);  // Earth
  
  // Calculate orbital period
  double TimePeriod = o_ref.get_TimePeriod();
  output_file << "Orbital period: " << TimePeriod << " seconds (" << TimePeriod/60.0 << " minutes)\n";
  std::cout << "Orbital period: " << TimePeriod << " seconds\n";
  
  // Create time vector: sample every 2.5 minutes (150 seconds) for 3.5 periods
  double sampling_interval = 150.0;  // seconds (2.5 minutes)
  double total_time = 3.5 * TimePeriod;
  int num_time_points = static_cast<int>(total_time / sampling_interval) + 1;
  
  Eigen::VectorXd times(num_time_points);
  for(int j = 0; j < num_time_points; ++j){
    times(j) = j * sampling_interval;
  }
  
  output_file << "Total propagation time: " << total_time << " seconds (" << total_time/TimePeriod << " periods)\n";
  output_file << "Sampling interval: " << sampling_interval << " seconds\n";
  output_file << "Number of time points: " << num_time_points << "\n";
  
  // Propagate using J2 with odeint (EOM_int=2 for J2)
  o_ref.propagate_2BP_odeint(times, 2, "sat_propagation_j2");
  output_file << "J2 propagation complete. Reading results...\n";
  
  // Read propagated results from CSV file
  std::ifstream prop_file("Results/sat_propagation_j2_file.csv");
  std::string header;
  std::getline(prop_file, header);  // Skip header
  
  int detection_count = 0;
  
  // Process each time point
  for(int j = 0; j < num_time_points; ++j){
    double current_time;
    double a_prop, e_prop, i_prop, RAAN_prop, AoP_prop, nu_prop;
    double energy_prop, rx, ry, rz, vx, vy, vz;
    double ax, ay, az, hx, hy, hz;
    char comma;
    
    prop_file >> current_time >> comma
              >> a_prop >> comma >> e_prop >> comma >> i_prop >> comma
              >> RAAN_prop >> comma >> AoP_prop >> comma >> nu_prop >> comma
              >> energy_prop >> comma
              >> rx >> comma >> ry >> comma >> rz >> comma
              >> vx >> comma >> vy >> comma >> vz >> comma
              >> ax >> comma >> ay >> comma >> az >> comma
              >> hx >> comma >> hy >> comma >> hz;
    
    output_file << "============================================\n";
    output_file << "Time: " << current_time << " s (Period: " << current_time/TimePeriod << ")\n";
    output_file << "Orbital Elements: a=" << a_prop << ", e=" << e_prop 
                << ", i=" << i_prop << ", RAAN=" << RAAN_prop 
                << ", AoP=" << AoP_prop << ", nu=" << nu_prop << "\n";
    
    // Create orbit object at this propagated state
    Orbit o;
    Eigen::Vector3d r_prop(rx, ry, rz);
    Eigen::Vector3d v_prop(vx, vy, vz);
    o.set_cartesian(r_prop, v_prop);
    o.set_mu(0);
    
    // Create Satellite Object
    Satellite sat;
    sat.set_orbit(o);
    sat.set_sensor_vectors(boom_angles);
    sat.set_soliton_params(soliton_params);
    
    output_file << "Satellite state: R=(" << rx << ", " << ry << ", " << rz 
                << "), V=(" << vx << ", " << vy << ", " << vz << ")\n";
    output_file << "Running Detection Simulation for " << num_samples << " debris samples...\n";
    
    sat.detection_sim(num_samples, search_radius, final_time);
    detection_count++;
    
    output_file << "Detection simulation complete for this time point.\n";
    output_file << "============================================\n\n";
  }
  
  prop_file.close();
  
  output_file << "All time points processed.\n";
  output_file << "Total detections across " << detection_count << " time points.\n";
  output_file << "J2 Propagation simulation complete.\n";
  output_file.close();

  std::cout << "Simulation complete!\n";
  
  // std::cout<< "\nStarting Detection Simulation for debris samples from file...\n";
  // sat.detection_sim("Results/Debris_20260501_101935/debris_samples_20260501_101935.csv", final_time);


  // auto sim_output = sat.detection_sim(num_samples, search_radius, final_time, soliton_params);
  // Eigen::MatrixXd debris_samples = sim_output.first;
  // std::vector<DetectionResult> detection_results = sim_output.second;

  // std::cout << "\n============================================\n";
  // std::cout << "Simulation Complete. Detections found: "
  //           << detection_results.size() << " out of " << num_samples << "\n";
  // std::cout << "============================================\n";

  // // Save to files in Results/ with timestamp
  // auto t = std::time(nullptr);
  // auto tm = *std::localtime(&t);
  // std::ostringstream oss_time;
  // oss_time << std::put_time(&tm, "%Y%m%d_%H%M%S");
  // std::string timestamp = oss_time.str();

  // std::string result_dir = "Results/Sim_" + timestamp;
  // std::string mkdir_cmd = "mkdir -p " + result_dir;
  // if (system(mkdir_cmd.c_str()) != 0) {
  //     std::cerr << "Failed to create directory: " << result_dir << "\n";
  // }

  // std::string filename_debris = result_dir + "/debris_samples_" + timestamp + ".csv";

  // std::ofstream outfile(filename_debris);
  // outfile << std::setprecision(15);
  // outfile << "x,y,z,vx,vy,vz\n";

  // // Save Satellite first
  // Orbit sat_o = sat.get_orbit();
  // Eigen::VectorXd sat_st = sat_o.get_cartesian();
  // outfile << sat_st(0) << "," << sat_st(1) << "," << sat_st(2) << ","
  //         << sat_st(3) << "," << sat_st(4) << "," << sat_st(5) << "\n";

  // // Save Debris Samples
  // for (int i = 0; i < debris_samples.rows(); ++i) {
  //   outfile << debris_samples(i, 0) << "," << debris_samples(i, 1) << ","
  //           << debris_samples(i, 2) << "," << debris_samples(i, 3) << ","
  //           << debris_samples(i, 4) << "," << debris_samples(i, 5) << "\n";
  // }
  // outfile.close();
  // std::cout << "Debris samples saved to " << filename_debris << "\n";

  // std::string filename_results = result_dir + "/detection_results_" + timestamp + ".csv";

  // std::ofstream resfile(filename_results);
  // resfile << std::setprecision(15);
  // resfile << "debris_id,detected,first_detection_time,sensor_hits\n";
  // for (const auto &res : detection_results) {
  //   resfile << res.debris_id << "," << (res.detected ? "true" : "false") << ","
  //           << res.first_detection_time << ",";
  //   for (size_t det_idx = 0; det_idx < res.detections.size(); ++det_idx) {
  //     resfile << "S" << res.detections[det_idx].sensor_id << "@"
  //             << res.detections[det_idx].time;
  //     if (det_idx < res.detections.size() - 1) {
  //       resfile << ";";
  //     }
  //   }
  //   resfile << "\n";
  // }
  // resfile.close();
  // std::cout << "Detection results saved to " << filename_results << "\n";

  // std::string filename_readme = result_dir + "/README.txt";
  // std::ofstream readmefile(filename_readme);
  // readmefile << "Simulation Timestamp: " << timestamp << "\n\n";
  
  // readmefile << "--- Satellite Orbit ---\n";
  // readmefile << "Initial OE [a_km, e, i_deg, omega_deg, Omega_deg, theta_deg]:\n";
  // readmefile << a << ", " << e << ", " << i << ", " << omega << ", " << Omega << ", " << theta << "\n\n";
  // readmefile << "Initial Cartesian State [x, y, z, vx, vy, vz]:\n" << sat_st.transpose() << "\n\n";

  // readmefile << "--- Detection Parameters ---\n";
  // readmefile << "Detection Frequency (Hz): " << sat.get_detection_freq() << "\n";
  // readmefile << "Wake Plane Angle (deg): " << (sat.get_wake_angle() * 180.0 / M_PI) << "\n\n";

  // readmefile << "--- Soliton Parameters ---\n";
  // readmefile << "Cone Angle (deg): " << (soliton_params[0] * 180.0 / M_PI) << "\n";
  // readmefile << "Cone Height (km): " << soliton_params[1] << "\n";
  // readmefile << "Velocity Multiplier: " << soliton_params[2] << "\n\n";

  // readmefile << "--- Sensor Vectors (Body Frame) ---\n";
  // const Eigen::Vector3d* sensors = sat.get_sensor_vectors();
  // for(int idx_s=0; idx_s<4; ++idx_s) {
  //     readmefile << "Sensor " << (idx_s+1) << ": [" << sensors[idx_s].transpose() << "]\n";
  // }
  // readmefile.close();
  // std::cout << "Simulation metadata saved to " << filename_readme << "\n";

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

  return 0;
}