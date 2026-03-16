#ifndef SATELLITE_H
#define SATELLITE_H

#include <eigen-5.0.0/Eigen/Dense>
#include <vector>
#include <utility>
#include "orbit.h"
#include "soliton.h"

struct DetectionEvent {
    int sensor_id;
    double time;
};

struct DetectionResult {
    int debris_id;
    bool detected;
    double first_detection_time;
    std::vector<DetectionEvent> detections;
};

class Satellite {
public:
    // constructor
    Satellite();

    // set functions
    // void set_soliton_state(Eigen::Vector3d pos, Eigen::Vector3d vol);
    /**
     * @brief Set the orbit of the satellite
     * @param orbit Orbit object
     */
    void set_orbit(Orbit orbit);

    /**
     * @brief Set the sensor vectors
     * @param angles Array of sensor angles in degrees [az1, el1, az2, el2, az3, el3, az4, el4]
     */
    void set_sensor_vectors(Eigen::ArrayXd angles);

    /**
     * @brief Set the wake angle
     * @param angle Angle in degrees
     */
    void set_wake_angle(double angle);

    // get functions

    /**
     * @brief Get the orbit of the satellite
     * @return Orbit object
     */
    Orbit get_orbit();

    /**
     * @brief check if a given postition vector in ECI frame is in the wake of satellite at the current time
     * @param pos Position of the point in ECI frame
     * @return True if the point is in the wake, false otherwise
     */
    bool within_wake(Eigen::Vector3d pos);  

    /**
     * @brief Generate debris samples around the satellite's current position, ensuring they are outside the wake.
     * @param num_samples Number of debris samples to generate.
     * @param search_radius_km Radius around the satellite to sample debris positions (km).
     * @return Matrix containing position and velocity of valid debris samples (num_samples x 6 matrix in ECI frame).
     */
    Eigen::MatrixXd generate_debris_samples(int num_samples, double search_radius_km);

    /**
     * @brief Simulation to generate random debris and check if the generated soliton is detected.
     */
    std::pair<Eigen::MatrixXd, std::vector<DetectionResult>> detection_sim(int no_of_samples, double search_radius_km, double final_time);

    /**
     * @brief Simulation to check if the generated soliton is detected from a pre-defined array.
     */
    std::vector<DetectionResult> detection_sim(Eigen::MatrixXd debris_samples, double final_time);

private:
    // Body frame: x: velocity, y: right, z: down
    // size (in body frame)
    // dimensions of the satellite
    double sat_x_size;
    double sat_y_size;
    double sat_z_size;

    // boom length for sensors
    double boom_length;
    double detection_freq; // in Hz

    // sensor angles in body frame
    double alpha_x, alpha_z;

    Eigen::Vector3d sensor_vectors[4];

    // sensor positions in body frame (mounted at corners of front yz face)
    Eigen::Vector3d sensor_1_BF, sensor_2_BF, sensor_3_BF, sensor_4_BF;
    Eigen::Vector3d corner_1_BF, corner_2_BF, corner_3_BF, corner_4_BF;

    // sensor positions in ECI frame
    Eigen::Vector3d sensor_1_ECI, sensor_2_ECI, sensor_3_ECI, sensor_4_ECI;


    Orbit sat_orbit; // satellite orbit

    // initial state vectors
    Eigen::Vector3d sat_R0; // in km
    Eigen::Vector3d sat_V0; // in km/s

    // current state vectors
    double time; // in seconds
    Eigen::Vector3d sat_R; // in km
    Eigen::Vector3d sat_V; // in km/s

    // Future state vectors
    Eigen::ArrayXXd future_state; // time, position (x,y,z), velocity (vx,vy,vz)
    
    // Body frame in ECI frame
    Eigen::Matrix3d M_BF_to_ECI; // rotation matrix from body frame to ECI frame

    /**
     * @brief Convert body frame to ECI frame using current position and velocity vectors
     * uses sat_R and sat_V to compute the rotation matrix
     */
    void BF_to_ECI();

    /**
     * @brief Convert a position vector from body frame to ECI frame
     * @param pos_BF Position vector in body frame
     * @return Position vector in ECI frame
     */
    Eigen::Vector3d pos_BF2ECI(Eigen::Vector3d pos_BF);

    /**
     * @brief Convert a position vector from ECI frame to body frame
     * @param pos_ECI Position vector in ECI frame
     * @return Position vector in body frame
     */
    Eigen::Vector3d pos_ECI2BF(Eigen::Vector3d pos_ECI);

    /**
     * @brief Convert a velocity vector from body frame to ECI frame
     * @param vel_BF Velocity vector in body frame
     * @return Velocity vector in ECI frame
     */
    Eigen::Vector3d vel_BF2ECI(Eigen::Vector3d vel_BF);

    /**
     * @brief Convert a velocity vector from ECI frame to body frame
     * @param vel_ECI Velocity vector in ECI frame
     * @return Velocity vector in body frame
     */
    Eigen::Vector3d vel_ECI2BF(Eigen::Vector3d vel_ECI);

    // // soliton
    // Soliton soliton;
    // double time_to_reach_cone_base;
    // Eigen::Vector3d debris_position, debris_velocity;

    // csv read
    void read_propagation_csv(const std::string &filename);
    void read_future_state(std::string name);

    // wake parameters
    double plane_angle; // in radians

    // plane parameters (BF)
    Eigen::Vector3d back_plane_normal_BF, plane_normal_BF_1, plane_normal_BF_2, plane_normal_BF_3, plane_normal_BF_4;
    double d_back_BF, d_1_BF, d_2_BF, d_3_BF, d_4_BF;
    
    // Detection functions
    double detections;

};

#endif // SATELLITE_H