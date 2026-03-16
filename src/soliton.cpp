#include "soliton.h"

// default constructor
Soliton::Soliton() {
    cone_angle = 45.0 * M_PI / 180; // radians
    cone_height = 10.0; // km
    sol_vel_multiplier = 1.2;
    debris_pos = Eigen::Vector3d(0.0, 0.0, 0.0);
    debris_vel = Eigen::Vector3d(7.0, 0.0, 0.0);
    soliton_velocity = debris_vel * sol_vel_multiplier;
    time_to_reach_cone_base = cone_height / soliton_velocity.norm(); // time to reach the apex of the cone
    time_of_generation = 0.0;
    shell_thickness = 10.0 * 1e-6; // 10 cm in km
}

// constructor
Soliton::Soliton(double angle, double height, double vel_multiplier, Eigen::Vector3d pos, Eigen::Vector3d vel) {
    cone_angle = angle;
    cone_height = height;
    sol_vel_multiplier = vel_multiplier;
    debris_pos = pos;
    debris_vel = vel;

    soliton_velocity = sol_vel_multiplier * debris_vel;
    time_to_reach_cone_base = cone_height / soliton_velocity.norm(); // time to reach the apex of the cone
    time_of_generation = 0.0;
    shell_thickness = 10.0 * 1e-6; // 10 cm in km
}

Soliton::Soliton(Eigen::Vector3d deb_pos, Eigen::Vector3d deb_vel, double generation_time) {
    cone_angle = 45.0 * M_PI / 180.0; // radians
    cone_height = 10.0; // km
    sol_vel_multiplier = 1.2;
    shell_thickness = 10.0 * 1e-6; // 10 cm in km

    debris_pos = deb_pos;
    debris_vel = deb_vel;
    time_of_generation = generation_time;
    soliton_velocity = deb_vel * sol_vel_multiplier;
    if (soliton_velocity.norm() > 0) {
        time_to_reach_cone_base = cone_height / soliton_velocity.norm();
    } else {
        time_to_reach_cone_base = 0;
    }
}

// set functions
void Soliton::set_params(double angle, double height, double vel_multiplier){
    cone_angle = angle;
    cone_height = height;
    sol_vel_multiplier = vel_multiplier;
    
    soliton_velocity = sol_vel_multiplier * debris_vel;
    time_to_reach_cone_base = cone_height / soliton_velocity.norm(); // time to reach the apex of the cone
}

void Soliton::set_debris_state(Eigen::Vector3d pos, Eigen::Vector3d vel) {
    debris_pos = pos;
    debris_vel = vel;
    soliton_velocity = sol_vel_multiplier * debris_vel;
    time_to_reach_cone_base = cone_height / soliton_velocity.norm(); // time to reach the apex of the cone
}



// get functions
Eigen::Vector3d Soliton::get_velocity() {
    return soliton_velocity;
}

double Soliton::get_time_to_reach_cone_base() {
    return time_to_reach_cone_base;
}



// detection functions

bool Soliton::within_cone(Eigen::Vector3d pos) {
    // debris position is the cone apex
    Eigen::Vector3d cone_apex = debris_pos;

    // position vector from apex
    Eigen::Vector3d debris_to_pos = pos - debris_pos;

    bool within_height = (debris_to_pos.norm() <= cone_height);
    bool within_angle = (debris_to_pos.dot(soliton_velocity) / (debris_to_pos.norm() * soliton_velocity.norm()) >= cos(cone_angle));
    bool same_side = (debris_to_pos.dot(soliton_velocity) >= 0);

    return (within_height && within_angle && same_side);
}

bool Soliton::within_spherical_range(Eigen::Vector3d pos, double time, double detection_freq) {
    Eigen::Vector3d debris_to_pos = pos - debris_pos;
    double range = time * soliton_velocity.norm();
    double del_range = 1/detection_freq * soliton_velocity.norm();
    return (debris_to_pos.norm() >= range - del_range && debris_to_pos.norm() <= range + del_range);
}

bool Soliton::within_spherical_shell(Eigen::Vector3d pos, double t) {
    double shell_radius = soliton_velocity.norm() * (t - time_of_generation);
    double apex_to_point = (pos - debris_pos).norm();
    return (apex_to_point >= (shell_radius - shell_thickness / 2.0)) && 
           (apex_to_point <= (shell_radius + shell_thickness / 2.0));
}

bool Soliton::within_soliton_shell(Eigen::Vector3d pos, double t) {
    return within_cone(pos) && within_spherical_shell(pos, t);
}