#include "satellite.h"
#include "constants.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif


// constructor
Satellite::Satellite() {
    // Define default satellite parameters

    // Body frame: x: velocity, y: right, z: down
    // size (in body frame)
    sat_x_size = 3.0*0.0001; // 30 centimeters (3U)
    sat_y_size = 2.0*0.0001;   // 20 centimeters (2U)
    sat_z_size = 2.0*0.0001;   // 20 centimeters (2U)

    // boom length for sensors
    boom_length = 0.1; // 1 meter
    detection_freq = Constants::DETECTION_FREQ; // 1000 Hz
    detections = 0;

    // sensor angles in body frame
    alpha_x = 54.7356 * Constants::PI / 180; // 54.7356 degrees
    alpha_z = 45.0 * Constants::PI / 180; // 45 degrees

    // sensor vectors in body frame
    sensor_vectors[0] = Eigen::Vector3d(std::cos(alpha_x), std::sin(alpha_z)*std::sin(alpha_x), std::cos(alpha_z)*std::sin(alpha_x));
    sensor_vectors[1] = Eigen::Vector3d(std::cos(alpha_x), -std::sin(alpha_z)*std::sin(alpha_x), std::cos(alpha_z)*std::sin(alpha_x));
    sensor_vectors[2] = Eigen::Vector3d(std::cos(alpha_x), -std::sin(alpha_z)*std::sin(alpha_x), -std::cos(alpha_z)*std::sin(alpha_x));
    sensor_vectors[3] = Eigen::Vector3d(std::cos(alpha_x), std::sin(alpha_z)*std::sin(alpha_x), -std::cos(alpha_z)*std::sin(alpha_x));

    // define a satellite body frame (x: velocity, y: right, z: down)
    corner_1_BF = Eigen::Vector3d(sat_x_size/2, sat_y_size/2, sat_z_size/2);
    corner_2_BF = Eigen::Vector3d(sat_x_size/2, -sat_y_size/2, sat_z_size/2);
    corner_3_BF = Eigen::Vector3d(sat_x_size/2, -sat_y_size/2, -sat_z_size/2);
    corner_4_BF = Eigen::Vector3d(sat_x_size/2, sat_y_size/2, -sat_z_size/2);

    sensor_1_BF = corner_1_BF + boom_length*sensor_vectors[0];
    sensor_2_BF = corner_2_BF + boom_length*sensor_vectors[1];
    sensor_3_BF = corner_3_BF + boom_length*sensor_vectors[2];
    sensor_4_BF = corner_4_BF + boom_length*sensor_vectors[3];

    // satellite orbit
    Eigen::VectorXd OE_sat(6);
    double a, e, i, omega, Omega, theta;
    a =         750 + 6371; // km;
    e =         0.063;
    i =         45;
    omega =     0;
    Omega =     0;
    theta =     0;

    OE_sat << a, e, i, omega, Omega, theta;
    sat_orbit.set_OE(OE_sat);

    // initial state vectors
    Eigen::VectorXd sat_state = sat_orbit.get_cartesian();
    sat_R0 = sat_state.segment(0,3); // in km
    sat_V0 = sat_state.segment(3,3); // in km/s

    // current state vectors
    time = 0.0; // initial time
    sat_R = sat_R0; // in km
    sat_V = sat_V0; // in km/s

    // Body frame in ECI frame
    BF_to_ECI();

    // wake parameters
    plane_angle = 60.0 * Constants::PI / 180; // radians
    
    back_plane_normal_BF << -1, 0, 0;
    d_back_BF = sat_x_size/2;

    plane_normal_BF_1 << -std::sin(plane_angle), 0, -std::cos(plane_angle);
    d_1_BF = sat_x_size/2 *std::sin(plane_angle) - sat_z_size/2 *std::cos(plane_angle);

    plane_normal_BF_2 << -std::sin(plane_angle), 0, std::cos(plane_angle);
    d_2_BF = sat_x_size/2 *std::sin(plane_angle) - sat_z_size/2 *std::cos(plane_angle);

    plane_normal_BF_3 << -std::sin(plane_angle), std::cos(plane_angle), 0;
    d_3_BF = sat_x_size/2 *std::sin(plane_angle) - sat_y_size/2 *std::cos(plane_angle);

    plane_normal_BF_4 << -std::sin(plane_angle), -std::cos(plane_angle), 0;
    d_4_BF = sat_x_size/2 *std::sin(plane_angle) - sat_y_size/2 *std::cos(plane_angle);

    // // soliton 
    // soliton.set_params(45.0 * M_PI / 180, 10, 1.2); // 45 deg cone angle, 10 km height, 1.2x debris velocity
}

// Define rotation matrix from body frame to ECI frame based on current satellite position and velocity
// and update all sensor positions in ECI frame
void Satellite::BF_to_ECI() {
    Eigen::Vector3d sat_x_BF;
    if (sat_V.norm() != 0) {
        sat_x_BF = sat_V.normalized();
    } else {
        sat_x_BF << 1.0, 0.0, 0.0;
    }

    Eigen::Vector3d sat_z_BF;
    if (sat_R.norm() != 0) {
        sat_z_BF = -sat_R.normalized();
    } else {
        sat_z_BF << 0.0, 0.0, 1.0;
    }

    Eigen::Vector3d sat_y_BF = sat_z_BF.cross(sat_x_BF);
    if (sat_y_BF.norm() != 0) {
        sat_y_BF.normalize();
    }

    M_BF_to_ECI.col(0) = sat_x_BF;
    M_BF_to_ECI.col(1) = sat_y_BF;
    M_BF_to_ECI.col(2) = sat_z_BF;

    // sensor positions in ECI frame
    sensor_1_ECI = sat_R + M_BF_to_ECI * sensor_1_BF;
    sensor_2_ECI = sat_R + M_BF_to_ECI * sensor_2_BF;
    sensor_3_ECI = sat_R + M_BF_to_ECI * sensor_3_BF;
    sensor_4_ECI = sat_R + M_BF_to_ECI * sensor_4_BF;
}

Eigen::Vector3d Satellite::pos_BF2ECI(Eigen::Vector3d pos_BF) {
    return sat_R + M_BF_to_ECI * pos_BF;
}

Eigen::Vector3d Satellite::pos_ECI2BF(Eigen::Vector3d pos_ECI) {
    return M_BF_to_ECI.transpose() * (pos_ECI - sat_R);
}

Eigen::Vector3d Satellite::vel_BF2ECI(Eigen::Vector3d vel_BF) {
    return M_BF_to_ECI * vel_BF;
}

Eigen::Vector3d Satellite::vel_ECI2BF(Eigen::Vector3d vel_ECI) {
    return M_BF_to_ECI.transpose() * vel_ECI;
}

// Set functions

// void Satellite::set_soliton_state(Eigen::Vector3d pos, Eigen::Vector3d vel) {
//     debris_position = pos;
//     debris_velocity = vel;
//     soliton.set_debris_state(debris_position, debris_velocity);
//     time_to_reach_cone_base = soliton.get_time_to_reach_cone_base();
//     std::cout<< "Soliton Velocity: " << soliton.get_velocity().transpose() << " km/s\n";
//     sat_orbit.propagate_2BP(1/detection_freq, time_to_reach_cone_base, 0, "satellite_propagation");

//     read_future_state("satellite_propagation");
//     detections = 0;
// }

void Satellite::set_orbit(Orbit orbit) {

    sat_orbit = orbit;
    sat_R0 = sat_orbit.get_cartesian().segment(0,3);
    sat_V0 = sat_orbit.get_cartesian().segment(3,3);
    // reset current position and velocity to initial values
    sat_R = sat_R0;
    sat_V = sat_V0;
    time = 0.0;
    BF_to_ECI();
}

void Satellite::set_sensor_vectors(Eigen::ArrayXd angles) {
    // Eigen::Vector3d(std::cos(alpha_x), std::sin(alpha_z)*std::sin(alpha_x), std::cos(alpha_z)*std::sin(alpha_x));
    sensor_vectors[0] = Eigen::Vector3d(std::cos(angles[0]), std::sin(angles[0])*std::sin(angles[1]), std::cos(angles[1])*std::sin(angles[0]));
    sensor_vectors[1] = Eigen::Vector3d(std::cos(angles[2]),-std::sin(angles[2])*std::sin(angles[3]), std::cos(angles[3])*std::sin(angles[2]));
    sensor_vectors[2] = Eigen::Vector3d(std::cos(angles[4]),-std::sin(angles[4])*std::sin(angles[5]),-std::cos(angles[5])*std::sin(angles[4]));
    sensor_vectors[3] = Eigen::Vector3d(std::cos(angles[6]), std::sin(angles[6])*std::sin(angles[7]),-std::cos(angles[7])*std::sin(angles[6]));
    sensor_1_BF = corner_1_BF + boom_length*sensor_vectors[0];
    sensor_2_BF = corner_2_BF + boom_length*sensor_vectors[1];
    sensor_3_BF = corner_3_BF + boom_length*sensor_vectors[2];
    sensor_4_BF = corner_4_BF + boom_length*sensor_vectors[3];
    BF_to_ECI();
}

void Satellite::set_wake_angle(double angle) {
    plane_angle = angle * Constants::PI / 180.0;
    // update plane parameters
    plane_normal_BF_1 << -std::sin(plane_angle), 0, -std::cos(plane_angle);
    d_1_BF = sat_x_size/2 *std::sin(plane_angle) - sat_z_size/2 *std::cos(plane_angle);

    plane_normal_BF_2 << -std::sin(plane_angle), 0, std::cos(plane_angle);
    d_2_BF = sat_x_size/2 *std::sin(plane_angle) - sat_z_size/2 *std::cos(plane_angle);

    plane_normal_BF_3 << -std::sin(plane_angle), std::cos(plane_angle), 0;
    d_3_BF = sat_x_size/2 *std::sin(plane_angle) - sat_y_size/2 *std::cos(plane_angle);

    plane_normal_BF_4 << -std::sin(plane_angle), -std::cos(plane_angle), 0;
    d_4_BF = sat_x_size/2 *std::sin(plane_angle) - sat_y_size/2 *std::cos(plane_angle);
}

// Get functions

Orbit Satellite::get_orbit() {
    return sat_orbit;
}


// check if a point is in the satellite wake
bool Satellite::within_wake(Eigen::Vector3d pos) {
    // position vector in body frame
    Eigen::Vector3d pos_BF = pos_ECI2BF(pos);
    
    bool behind_back_plane, behind_plane_1, behind_plane_2, behind_plane_3, behind_plane_4;
    behind_back_plane = (back_plane_normal_BF.dot(pos_BF) - d_back_BF) >= 0;
    behind_plane_1 = (plane_normal_BF_1.dot(pos_BF) - d_1_BF) >= 0;
    behind_plane_2 = (plane_normal_BF_2.dot(pos_BF) - d_2_BF) >= 0;
    behind_plane_3 = (plane_normal_BF_3.dot(pos_BF) - d_3_BF) >= 0;
    behind_plane_4 = (plane_normal_BF_4.dot(pos_BF) - d_4_BF) >= 0;

    return behind_back_plane && behind_plane_1 && behind_plane_2 && behind_plane_3 && behind_plane_4;
}

Eigen::MatrixXd Satellite::generate_debris_samples(int num_samples, double search_radius_km, int num_headings) {
    int total_samples = num_samples * num_headings;
    Eigen::MatrixXd samples(total_samples, 6);

    // LVLH base vectors from satellite current R and V
    Eigen::Vector3d h_vec = sat_R.cross(sat_V);
    Eigen::Vector3d u_radial = sat_R.normalized();
    Eigen::Vector3d u_cross = h_vec.normalized();
    Eigen::Vector3d u_along = u_cross.cross(u_radial);

    #pragma omp parallel
    {
        // Thread-local random number generation
        std::random_device rd;
        // Seed the generator with the thread ID to ensure distinct sequences
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        std::mt19937 gen(rd() ^ thread_id);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::normal_distribution<double> normal(0.0, 0.1);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_samples; ++i) {
            bool valid = false;

            Eigen::Vector3d offset_BF;

            while (!valid) {
                // Generate random offsets in BODY FRAME
                double u = uniform(gen);
                double v = uniform(gen);
                double theta = 2.0 * Constants::PI * u;
                double phi = std::acos(2.0 * v - 1.0);
                double r = search_radius_km * std::cbrt(uniform(gen));

                // Convert to Cartesian (Body Frame)
                double x = r * std::sin(phi) * std::cos(theta);
                double y = r * std::sin(phi) * std::sin(theta);
                double z = r * std::cos(phi);

                offset_BF << x, y, z;

                // Wake check
                bool check_back = (back_plane_normal_BF.dot(offset_BF) - d_back_BF) >= 0;
                bool check_1 = (plane_normal_BF_1.dot(offset_BF) - d_1_BF) >= 0;
                bool check_2 = (plane_normal_BF_2.dot(offset_BF) - d_2_BF) >= 0;
                bool check_3 = (plane_normal_BF_3.dot(offset_BF) - d_3_BF) >= 0;
                bool check_4 = (plane_normal_BF_4.dot(offset_BF) - d_4_BF) >= 0;

                bool in_wake = check_back && check_1 && check_2 && check_3 && check_4;

                if (!in_wake) { valid = true;}
            }


            // Convert to ECI
            Eigen::Vector3d pos_ECI = pos_BF2ECI(offset_BF);

            // Position magnitude
            double r_mag = pos_ECI.norm();
            // Exact circular velocity magnitude
            double v_mag = std::sqrt(Constants::GM_EARTH / r_mag);

            // Define the local radial unit vector
            Eigen::Vector3d u_radial = pos_ECI.normalized();

            // (Any arbitrary perpendicular vector works as a starting point)
            Eigen::Vector3d arbitrary(0, 1, 0); 
            if (std::abs(u_radial.dot(arbitrary)) > 0.99) arbitrary = Eigen::Vector3d(1, 0, 0);

            // Create the Local Horizontal Plane (u_east, u_north)
            Eigen::Vector3d u_east = u_radial.cross(arbitrary).normalized();
            Eigen::Vector3d u_north = u_radial.cross(u_east).normalized();
            
            // int num_headings = 72; // e.g., sample every 5 degrees
            for (int h = 0; h < num_headings; ++h) {
                double heading = (h * 2.0 * Constants::PI) / num_headings;

                // Circular velocity vector in ECI
                Eigen::Vector3d v_eci = v_mag * (std::cos(heading) * u_east + std::sin(heading) * u_north);

                // Add tiny normal noise if desired for numerical stability
                v_eci(0) += normal(gen);
                v_eci(1) += normal(gen);
                v_eci(2) += normal(gen);
                
                // Write to the output matrix (Eigen is thread-safe for writing to distinct rows)
                samples.row(i*num_headings+h).segment<3>(0) = pos_ECI;
                samples.row(i*num_headings+h).segment<3>(3) = v_eci;
            
            }
        }
    }
    
    return samples;
}

// // check if the soliton is detected at current time
// bool Satellite::detect_soliton() {
//     bool detected = false;
//     bool debris_within_wake = within_wake(debris_position);
//     if (debris_within_wake) {
//         // std::cout<< "Debris is within satellite wake at time " << time << " seconds.\n";
//         // return false;
//     }
//     else{
//         // check each sensor
//         bool sensor_1_detected = soliton.within_cone(sensor_1_ECI) && soliton.within_spherical_range(sensor_1_ECI, time, detection_freq);
//         bool sensor_2_detected = soliton.within_cone(sensor_2_ECI) && soliton.within_spherical_range(sensor_2_ECI, time, detection_freq);
//         bool sensor_3_detected = soliton.within_cone(sensor_3_ECI) && soliton.within_spherical_range(sensor_3_ECI, time, detection_freq);
//         bool sensor_4_detected = soliton.within_cone(sensor_4_ECI) && soliton.within_spherical_range(sensor_4_ECI, time, detection_freq);

//         if (sensor_1_detected) {
//             std::cout << "Sensor 1 detected soliton at time " << time << " seconds.\n";
//             detections += 1;
//         }
//         if (sensor_2_detected) {
//             std::cout << "Sensor 2 detected soliton at time " << time << " seconds.\n";
//             detections += 1;
//         }
//         if (sensor_3_detected) {
//             std::cout << "Sensor 3 detected soliton at time " << time << " seconds.\n";
//             detections += 1;
//         }
//         if (sensor_4_detected) {
//             std::cout << "Sensor 4 detected soliton at time " << time << " seconds.\n";
//             detections += 1;
//         }
//         detected = sensor_1_detected || sensor_2_detected || sensor_3_detected || sensor_4_detected;
//     }
//     return detected;
// }

// check if soliton is detected at any time in future_state
// bool Satellite::detect_soliton_over_time() {
    
//     for (int i = 0; i < future_state.rows(); ++i) {
//         time = future_state(i, 0);
//         Eigen::Vector3d pos= future_state.row(i).segment<3>(8);
//         Eigen::Vector3d vel= future_state.row(i).segment<3>(11);
//         position = pos;
//         velocity = vel;
//         // update sensor positions in ECI frame
        
//         BF_to_ECI();

//         bool local_detected = detect_soliton();
//     }
//     std::cout << "Total Detections: " << detections << "\n";
//     if (detections>=2){return true;}
//     else {return false;}
// }

// Read a CSV file produced by orbit::propagate_2BP (header + numeric rows)
void Satellite::read_propagation_csv(const std::string &filename) {
    std::ifstream ifs(filename.c_str());
    if (!ifs.is_open()){
        std::cerr << "Satellite::read_propagation_csv: failed to open '" << filename << "'\n";
        return;
    }

    std::string line;
    // read header
    if (!std::getline(ifs, line)){
        std::cerr << "Satellite::read_propagation_csv: file empty: '" << filename << "'\n";
        return;
    }

    std::vector<std::vector<double>> rows;

    while (std::getline(ifs, line)){
        if (line.size() == 0) continue;
        std::vector<double> values;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')){
            // trim
            size_t start = cell.find_first_not_of(" \t\r\n");
            size_t end = cell.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) { cell = ""; }
            else cell = cell.substr(start, end - start + 1);

            if (cell.empty()) { values.push_back(0.0); continue; }

            try {
                double v = std::stod(cell);
                values.push_back(v);
            } catch (...) {
                // filter non-numeric characters
                std::string filtered;
                for (char c: cell) if ((c>='0' && c<='9') || c=='-' || c=='+' || c=='.' || c=='e' || c=='E') filtered.push_back(c);
                if (!filtered.empty()){
                    try { values.push_back(std::stod(filtered)); }
                    catch(...) { values.push_back(0.0); }
                } else {
                    values.push_back(0.0);
                }
            }
        }
        if (!values.empty()) rows.push_back(values);
    }

    if (rows.empty()) {
        std::cerr << "Debris::read_propagation_csv: no data rows found in '" << filename << "'\n";
        return;
    }

    size_t cols = 0;
    for (auto &r: rows) if (r.size() > cols) cols = r.size();

    Eigen::ArrayXXd out(rows.size(), cols);
    out.setZero();
    for (size_t i=0;i<rows.size();++i){
        for (size_t j=0;j<rows[i].size();++j) out(i,j) = rows[i][j];
    }

    future_state = out;
    return;
}

// Read satellite propagation file (tries "name_file.csv" then "name") and
// populate internal `future_state` matrix using read_propagation_csv.
void Satellite::read_future_state(std::string name) {
    std::string fn1 = "Results/" + name + "_file.csv";
    std::string fn2 = "Results/" + name;

    // try first filename
    read_propagation_csv(fn1);
    if (future_state.size() == 0) {
        // try second filename
        read_propagation_csv(fn2);
    }

    if (future_state.size() == 0) {
        std::cerr << "Satellite::read_future_state: failed to read '" << fn1 << "' or '" << fn2 << "'\n";
        return;
    }

    // Basic validation
    if (future_state.rows() < 1 || future_state.cols() < 14) {
        std::cerr << "Satellite::read_future_state: unexpected CSV layout (rows=" << future_state.rows() << ", cols=" << future_state.cols() << ")\n";
        // still keep future_state as read, but caller should handle it
    }

    return;
}

std::pair<Eigen::MatrixXd, std::vector<DetectionResult>> Satellite::detection_sim(int no_of_samples, double search_radius_km, double final_time) {
    Eigen::MatrixXd debris_samples = generate_debris_samples(no_of_samples, search_radius_km);
    std::vector<DetectionResult> results = detection_sim(debris_samples, final_time);
    return {debris_samples, results};
}

std::vector<DetectionResult> Satellite::detection_sim(Eigen::MatrixXd debris_samples, double final_time) {
    std::vector<DetectionResult> results;

    std::cout << "Propagating satellite orbit up to " << final_time << " seconds...\n";
    sat_orbit.propagate_2BP(1.0 / detection_freq, final_time, 2, "satellite_propagation");

    std::cout << "Reading future state from propagation output...\n";
    read_future_state("satellite_propagation");

    int num_steps = future_state.rows();
    if (num_steps == 0) {
        std::cerr << "Propagation state empty!\n";
        return results;
    }

    Eigen::ArrayXXd time_array = future_state.col(0);
    std::cout << "Simulating detections over " << num_steps << " timesteps for " << debris_samples.rows() << " samples...\n";

    #pragma omp parallel
    {
        std::vector<DetectionResult> local_results;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < debris_samples.rows(); ++i) {
            Eigen::Vector3d debris_pos = debris_samples.row(i).segment<3>(0);
            Eigen::Vector3d debris_vel = debris_samples.row(i).segment<3>(3);

            Soliton soliton(debris_pos, debris_vel, 0.0);
            Eigen::Vector3d sol_vel = soliton.get_velocity();

            int cadence = std::max(1, (int)detection_freq); 
            double min_dist = 1e9;
            int t_idx_closest = 0;

            for (int t_idx = 0; t_idx < num_steps; t_idx += cadence) {
                Eigen::Vector3d sat_pos = future_state.row(t_idx).segment<3>(8);
                double current_time = time_array(t_idx);
                Eigen::Vector3d sol_center = debris_pos + sol_vel * current_time;
                double dist = (sat_pos - sol_center).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    t_idx_closest = t_idx;
                }
            }

            int search_window_steps = cadence;
            int start_idx = std::max(0, t_idx_closest - search_window_steps);
            int end_idx = std::min(num_steps, t_idx_closest + search_window_steps);

            for (int t_idx = start_idx; t_idx < end_idx; ++t_idx) {
                Eigen::Vector3d sat_pos = future_state.row(t_idx).segment<3>(8);
                double current_time = time_array(t_idx);
                Eigen::Vector3d sol_center = debris_pos + sol_vel * current_time;
                double dist = (sat_pos - sol_center).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }

            if (min_dist > 12.0) {
                continue;
            }

            bool detected = false;
            double first_detection_time = -1.0;
            std::vector<DetectionEvent> detections;

            for (int t_idx = start_idx; t_idx < end_idx; ++t_idx) {
                double current_time = time_array(t_idx);
                Eigen::Vector3d sat_pos = future_state.row(t_idx).segment<3>(8);
                Eigen::Vector3d sat_vel = future_state.row(t_idx).segment<3>(11);

                Eigen::Vector3d sat_x_BF = (sat_vel.norm() != 0) ? sat_vel.normalized() : Eigen::Vector3d(1.0, 0.0, 0.0);
                Eigen::Vector3d sat_z_BF = (sat_pos.norm() != 0) ? (-sat_pos).normalized() : Eigen::Vector3d(0.0, 0.0, 1.0);
                Eigen::Vector3d sat_y_BF = sat_z_BF.cross(sat_x_BF);
                if (sat_y_BF.norm() != 0) {
                    sat_y_BF.normalize();
                }

                Eigen::Matrix3d R_mat;
                R_mat.col(0) = sat_x_BF;
                R_mat.col(1) = sat_y_BF;
                R_mat.col(2) = sat_z_BF;

                Eigen::Vector3d sensors_BF[4] = {sensor_1_BF, sensor_2_BF, sensor_3_BF, sensor_4_BF};

                for (int sensor_id = 0; sensor_id < 4; ++sensor_id) {
                    Eigen::Vector3d sensor_pos_ECI = R_mat * sensors_BF[sensor_id] + sat_pos;
                    if (soliton.within_soliton_shell(sensor_pos_ECI, current_time)) {
                        detections.push_back({sensor_id, current_time});
                        if (!detected) {
                            detected = true;
                            first_detection_time = current_time;
                        }
                    }
                }
            }

            if (detected) {
                DetectionResult res;
                res.debris_id = i;
                res.detected = detected;
                res.first_detection_time = first_detection_time;
                res.detections = detections;
                local_results.push_back(res);
            }
        }
        
        #pragma omp critical
        {
            results.insert(results.end(), local_results.begin(), local_results.end());
        }
    }
    return results;
}