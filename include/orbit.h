#ifndef ORBIT_H
#define ORBIT_H
#include <eigen-5.0.0/Eigen/Dense>

class Orbit{
	public:
		//set and print functions
		void set_OE(Eigen::VectorXd OE);
		void print_OE();
		void set_cartesian(Eigen::Vector3d r, Eigen::Vector3d v);
		void set_thrust(double aT);
		// 0 for Earth, 1 for Sun, 2 for dimensionless
		void set_mu(int i);
		void print_cartesian();
		void print_perifocal();
		void print_rotMat();
		//get functions
		double get_TimePeriod();
		Eigen::VectorXd get_cartesian();
		Eigen::VectorXd get_OE();
		// OE & cartesion conversions
		void OE_to_cartesian();
		void cartesian_to_OE();
		//propagate 2 body problem function (step size, final time, 
		// which EOM to be used (0 for ideal, 1 for thrust in velocity direction),
		// name of the file with final data)
		void propagate_2BP(double step, double trange, int EOM_int, std::string name);
		
		// propagate using boost odeint (time points, which EOM to be used,
		// name of the file with final data)
		// Automatically chooses internal time step; returns results at specified times
		void propagate_2BP_odeint(Eigen::VectorXd times, int EOM_int, std::string name);
		
		// Keplers problem
		Eigen::VectorXd Kepler_prob(Eigen::Vector3d R0, Eigen::Vector3d V0, double del_t);

		// equation of motion 
		Eigen::VectorXd EoM(Eigen::Vector3d r, Eigen::Vector3d v, int EOM_int);
	private:
		// Time Period
		double TimePeriod;
		//semi-major axis (km), eccentricity, inclination, RAAN, argument of periapsis and true-anomaly (rad)
        double a, e, i, RAAN, AoP, nu;
		// thrust value (kN/Kg)
		double thrust;
		//Cartesian Coordinates
		Eigen::Vector3d R;
		Eigen::Vector3d V;
		//perifocal coordinates
    	Eigen::Vector3d Rp;
    	Eigen::Vector3d Vp;
		//Rotation Matrices
		Eigen::Matrix3d Rctop;
		Eigen::Matrix3d Rptoc;
		//gravitational parameters
		double mu_E=398600.4418; //km^3/s^2
		double J2_E=0.00108248; 
		double mu_S=132712440018; //km^3/s^2
		double mu_dimensionless=1; 
		double RE=6371; //km
		double mu=mu_E;
		//Private function to covert any cartesian coordinates to Orbital Elements
		Eigen::ArrayXXd cartesian_to_OE_I(Eigen::Vector3d x, Eigen::Vector3d y);
		//function to get angular momentum
		Eigen::Vector3d angular_mom(Eigen::Vector3d x, Eigen::Vector3d y);
		//function to get orbital energy
		double orbital_energy(Eigen::Vector3d x, Eigen::Vector3d y, int EOM_int);
		void set_TimePeriod();
		// equation of motion of 2 body problem
		Eigen::VectorXd EoM_2BP(Eigen::Vector3d r, Eigen::Vector3d v);
		// equation of motion of 2BP with thrust in velocity direction
		Eigen::VectorXd EoM_2BP_thrust_V(Eigen::Vector3d r, Eigen::Vector3d v);
		// equation of motion of 2BP with J2 perturbations
		Eigen::VectorXd EoM_2BP_J2(Eigen::Vector3d R, Eigen::Vector3d V);
};

Eigen::ArrayXXd lambert(Eigen::Vector3d R0,Eigen::Vector3d Rf,double TOF, int DM);

#endif