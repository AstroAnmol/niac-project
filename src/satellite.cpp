#include "satellite.h"
#include "constants.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cmath>
#include <queue>
#include <string>
#include <vector>
#include <iomanip>
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
    boom_length = 0.001; // 1 meter
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

    // soliton 

    soliton_params << 10.0 * M_PI / 180, 10, 1.2; // 10 deg cone angle, 10 km height, 1.2x debris velocity
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

void Satellite::set_soliton_params(Eigen::Vector3d params) {
    soliton_params = params;
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

Eigen::MatrixXd Satellite::read_debris_csv(const std::string &filename) {
    std::ifstream ifs(filename.c_str());
    if (!ifs.is_open()){
        std::cerr << "Satellite::read_debris_csv: failed to open '" << filename << "'\n";
        return Eigen::MatrixXd();
    }

    std::string line;
    // read and skip header
    if (!std::getline(ifs, line)){
        std::cerr << "Satellite::read_debris_csv: file empty: '" << filename << "'\n";
        return Eigen::MatrixXd();
    }

    std::vector<std::vector<double>> rows;

    while (std::getline(ifs, line)){
        if (line.size() == 0) continue;
        std::vector<double> values;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')){
            // trim whitespace
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
                for (char c: cell) 
                    if ((c>='0' && c<='9') || c=='-' || c=='+' || c=='.' || c=='e' || c=='E')
                        filtered += c;
                
                if (!filtered.empty()){
                    try {
                        values.push_back(std::stod(filtered));
                    }
                    catch(...) {
                        values.push_back(0.0);
                    }
                } else {
                    values.push_back(0.0);
                }
            }
        }
        if (!values.empty()) rows.push_back(values);
    }

    if (rows.empty()) {
        std::cerr << "Satellite::read_debris_csv: no data rows found in '" << filename << "'\n";
        return Eigen::MatrixXd();
    }

    // Determine number of columns (should be 6: x, y, z, vx, vy, vz)
    size_t cols = 0;
    for (auto &r: rows) if (r.size() > cols) cols = r.size();

    // Create output matrix
    Eigen::MatrixXd debris_data(rows.size(), cols);
    debris_data.setZero();
    
    for (size_t i=0; i<rows.size(); ++i){
        for (size_t j=0; j<rows[i].size(); ++j) {
            debris_data(i, j) = rows[i][j];
        }
    }

    std::cout << "Successfully read " << rows.size() << " debris samples from '" << filename << "'\n";
    return debris_data;
}

void Satellite::detection_sim(int no_of_samples, double search_radius_km, double final_time) {
    Eigen::MatrixXd debris_samples = generate_debris_samples(no_of_samples, search_radius_km);

    // Save to files in Debris/ with timestamp
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss_time;
    oss_time << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss_time.str();

    std::string debris_dir = "Results/Debris_" + timestamp;
    std::string mkdir_cmd = "mkdir -p " + debris_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        std::cerr << "Failed to create directory: " << debris_dir << "\n";
    }

    std::string filename_debris = debris_dir + "/debris_samples_" + timestamp + ".csv";

    std::ofstream outfile(filename_debris);
    outfile << std::setprecision(15);
    outfile << "x,y,z,vx,vy,vz\n";

    // Save Satellite first
    Eigen::VectorXd sat_st = sat_orbit.get_cartesian();
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

    std::vector<DetectionResult> results = detection_sim(debris_samples, final_time);
    std::cout << "\n============================================\n";
    std::cout << "Simulation Complete. Detections found: "
              << results.size() << " out of " << debris_samples.rows()-1 << "\n";
    std::cout << "============================================\n";
    save_detection_results(results, filename_debris);
    // return {debris_samples, results};
    return;
}

void Satellite::detection_sim(const std::string& debris_filename, double final_time) {
    Eigen::MatrixXd debris_samples;
    debris_samples = read_debris_csv(debris_filename);
    if (debris_samples.size() == 0) {
        std::cerr << "Failed to read debris samples from '" << debris_filename << "'\n";
        return;
    }
    
    if (debris_samples.rows() < 2 || debris_samples.cols() < 6) {
        std::cerr << "Debris file '" << debris_filename << "' has insufficient data (needs at least 2 rows, 6 columns)\n";
        return;
    }
    
    // Print info about the satellite state stored in the debris file
    Eigen::VectorXd sat_st_original = debris_samples.row(0);
    std::cout << "Debris file was generated with satellite state: [" << sat_st_original.transpose() << "]\n";
    
    // Remove the first row which contains the satellite state
    Eigen::MatrixXd debris_samples_trimmed = debris_samples.bottomRows(debris_samples.rows() - 1).eval();

    std::vector<DetectionResult> results = detection_sim(debris_samples_trimmed, final_time);
    std::cout << "\n============================================\n";
    std::cout << "Simulation Complete. Detections found: "
              << results.size() << " out of " << debris_samples_trimmed.rows() << "\n";
    std::cout << "============================================\n";
    save_detection_results(results, debris_filename);
    return;
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
    
    if (debris_samples.cols() < 6) {
        std::cerr << "Error: Debris samples must have at least 6 columns (x,y,z,vx,vy,vz), but has " << debris_samples.cols() << "\n";
        return results;
    }

    #pragma omp parallel
    {
        std::vector<DetectionResult> local_results;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < debris_samples.rows(); ++i) {
            Eigen::Vector3d debris_pos = debris_samples.row(i).segment<3>(0);
            Eigen::Vector3d debris_vel = debris_samples.row(i).segment<3>(3);

            // soliton parameters: cone angle (radians), cone height (km), velocity multiplier 
            Soliton soliton(soliton_params[0], soliton_params[1], soliton_params[2], debris_pos, debris_vel);
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

void Satellite::save_detection_results(const std::vector<DetectionResult>& results, const std::string& debris_filename) {
    // Save results in Results/ with timestamp
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss_time;
    oss_time << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss_time.str();

    // make timestep directory
    std::string result_dir = "Results/Sim_" + timestamp;
    std::string mkdir_cmd = "mkdir -p " + result_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        std::cerr << "Failed to create directory: " << result_dir << "\n";
    }

    // Save results to CSV
    std::string filename_results = result_dir + "/detection_results_" + timestamp + ".csv";

    std::ofstream resfile(filename_results);
    resfile << std::setprecision(15);
    resfile << "debris_id,detected,first_detection_time,sensor_hits\n";
    for (const auto &res : results) {
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

    // Save a readme file with simulation parameters
    std::string filename_readme = result_dir + "/README.txt";
    std::ofstream readmefile(filename_readme);
    readmefile << "Simulation Timestamp: " << timestamp << "\n\n";
    
    readmefile << "--- Satellite Orbit ---\n";
    readmefile << "Initial OE [a_km, e, i_deg, omega_deg, Omega_deg, theta_deg]:\n";
    Eigen::VectorXd oe = sat_orbit.get_OE();
    readmefile << oe(0) << ", " << oe(1) << ", " << oe(2) << ", " << oe(3) << ", " << oe(4) << ", " << oe(5) << "\n\n";
    Eigen::VectorXd sat_st = sat_orbit.get_cartesian();
    readmefile << "Initial Cartesian State [x, y, z, vx, vy, vz]:\n" << sat_st.transpose() << "\n\n";

    readmefile << "--- Detection Parameters ---\n";
    readmefile << "Detection Frequency (Hz): " << detection_freq << "\n";
    readmefile << "Wake Plane Angle (deg): " << (plane_angle * 180.0 / M_PI) << "\n\n";
    readmefile << "Debris file used for detection sim: " << debris_filename << "\n\n";

    readmefile << "--- Soliton Parameters ---\n";
    readmefile << "Cone Angle (deg): " << (soliton_params[0] * 180.0 / M_PI) << "\n";
    readmefile << "Cone Height (km): " << soliton_params[1] << "\n";
    readmefile << "Velocity Multiplier: " << soliton_params[2] << "\n\n";

    readmefile << "--- Sensor Positions (Body Frame) ---\n";
    Eigen::Vector3d sensors_BF[4] = {sensor_1_BF, sensor_2_BF, sensor_3_BF, sensor_4_BF};
    for(int idx_s=0; idx_s<4; ++idx_s) {
        readmefile << "Sensor " << (idx_s+1) << ": [" << sensors_BF[idx_s].transpose() << "]\n";
    }
    readmefile.close();
    std::cout << "Simulation metadata saved to " << filename_readme << "\n";
}