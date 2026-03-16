#ifndef SOLITON_H
#define SOLITON_H

#include <eigen-5.0.0/Eigen/Dense>

class Soliton {
public:
    // constructor
    Soliton();
    Soliton(double angle, double height, double vel_multiplier, Eigen::Vector3d pos, Eigen::Vector3d vel);
    Soliton(Eigen::Vector3d deb_pos, Eigen::Vector3d deb_vel, double time_of_generation);
    
    // set functions
    void set_params(double angle, double height, double vel_multiplier);
    void set_debris_state(Eigen::Vector3d pos, Eigen::Vector3d vel);

    // get functions
    Eigen::Vector3d get_velocity();
    double get_time_to_reach_cone_base();

    // check within cone
    bool within_cone(Eigen::Vector3d pos);

    // check within spherical detection range
    bool within_spherical_range(Eigen::Vector3d pos, double time, double detection_freq);

    // check within spherical shell at time t
    bool within_spherical_shell(Eigen::Vector3d pos, double t);

    // check within soliton shell (intersection of cone and spherical shell)
    bool within_soliton_shell(Eigen::Vector3d pos, double t);

private:
    double cone_angle; // in radians
    double cone_height; // in km
    double sol_vel_multiplier;
    double time_to_reach_cone_base; // in seconds
    double time_of_generation; // in seconds
    double shell_thickness; // in km
    Eigen::Vector3d soliton_velocity; // in km/s

    Eigen::Vector3d debris_pos; // debris position when soliton is generated
    Eigen::Vector3d debris_vel; // debris velocity when soliton is generated

};

#endif // SOLITON_H