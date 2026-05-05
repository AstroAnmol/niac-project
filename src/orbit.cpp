#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include "orbit.h"


double stumpff_C(double z){
    if (z>1e-6){
        return (1- std::cos(std::sqrt(z)))/z;
    }
    else if (z<-1e-6){
        return (std::cosh(std::sqrt(-z))-1)/(-z);
    }
    else
    {
        return 0.5;
    }
}
double stumpff_S(double z){
    if (z>1e-6){
        double sz=std::sqrt(z);
        return (sz -std::sin(sz))/(std::pow(sz,3));
    }
    else if (z<-1e-6){
        double sz=std::sqrt(-z);
        return (std::sinh(sz)-sz)/(std::pow(sz,3));
    }
    else{
        return 1.0/6.0;
    }
    
}

Eigen::ArrayXXd lambert(Eigen::Vector3d R0,Eigen::Vector3d Rf,double TOF, int DM){
    long double r0, rf;
    long double cos_del_nu, A;
    long double psi, psi_up, psi_down, c2, c3;
    long double  del_t, y, ki;
    long double mu=132712440018; //km^3/s^2
    long double f, g, g_dot;
    Eigen::Vector3d V0, Vf;
    Eigen::ArrayXXd V(3,2);
    Eigen::ArrayXXd V1(3,2);
    V1<<0,0,0,0,0,0;
    r0=R0.norm();
    rf=Rf.norm();
    cos_del_nu=R0.dot(Rf)/(r0*rf);
    A=DM*std::sqrt(r0*rf*(1+cos_del_nu));
    if (A==0){
        std::cout<< "Error"<<std::endl;
        return V;
    }   
    else{
        psi=0;
        c2=1.0/2.0;
        c3=1.0/6.0;
        psi_up=4*M_PI*M_PI;
        psi_down=-4*M_PI;
        del_t=0;
        int maxiter=1000;
        int i=1;
        while (std::abs(TOF-del_t)>1e-6){
            if (i>maxiter){
                std::cout<<"error"<<std::endl;
                return V1;
            }
            y= r0 + rf + A*(psi*c3-1)/(std::sqrt(c2));
            ki=std::sqrt(y/c2);
            del_t=(ki*ki*ki*c3 + A*std::sqrt(y))/std::sqrt(mu);
            if (del_t<TOF){
                psi_down=psi;
            }
            else{
                psi_up=psi;
            }
            psi=(psi_up+psi_down)/2.0;
            c2=stumpff_C(psi);
            c3=stumpff_S(psi);
            i++;
        }
    }
    f=1-y/r0;
    g=A*std::sqrt(y/mu);
    g_dot=1-y/rf;

    V0=(Rf-f*R0)/g;
    Vf=(g_dot*Rf-R0)/g;
    V.col(0)=V0;
    V.col(1)=Vf;
    return V;
}

//set time period
void Orbit::set_TimePeriod(){
    TimePeriod=2*M_PI*std::sqrt(std::pow(a,3)/mu);
}

//set mu
void Orbit::set_mu(int i){
    if (i==0){
        mu=mu_E;
    }
    if (i==1){
        mu=mu_S;
    }
    if (i==2){
        mu=mu_dimensionless;
    }
}

//set thrust
void Orbit::set_thrust(double aT){
    thrust=aT;
}


//get time period
double Orbit::get_TimePeriod(){
    return TimePeriod;
}

//set the values of Orbital Elements, km, degrees
void Orbit::set_OE(Eigen::VectorXd OE){
    a= OE(0);
    e= OE(1);
    i= OE(2)*M_PI/180.0;
    RAAN= OE(3)*M_PI/180.0;
    AoP= OE(4)*M_PI/180.0;
    nu= OE(5)*M_PI/180.0;
    OE_to_cartesian();
    set_TimePeriod();
}



//print Orbital Elements, km, degrees
void Orbit::print_OE(){
    std::cout << "semi-major axis (km): " << a << std::endl;
    std::cout << "eccentricity: " << e << std::endl;
    std::cout << "inclination (deg): " << i*180/M_PI << std::endl;
    std::cout << "RAAN (deg): " << RAAN*180/M_PI << std::endl;
    std::cout << "argument of periapsis (deg): " << AoP*180/M_PI << std::endl;
    std::cout << "true anomaly (deg): " << nu*180/M_PI << std::endl;
}

//get cartesian coordinates in km, km/s
Eigen::VectorXd Orbit::get_cartesian(){
    Eigen::VectorXd cartesian(6);
    cartesian.block(0,0,3,1)=R;
    cartesian.block(3,0,3,1)=V;
    return cartesian;
}

//get orbital elements (km, degrees)
Eigen::VectorXd Orbit::get_OE(){
    Eigen::VectorXd OE(6);
    OE(0)=a;
    OE(1)=e;
    OE(2)=i*180/M_PI;
    OE(3)=RAAN*180/M_PI;
    OE(4)=AoP*180/M_PI;
    OE(5)=nu*180/M_PI;
    return OE;
}


//set the values of cartesian coordinates km, km/s
void Orbit::set_cartesian(Eigen::Vector3d r, Eigen::Vector3d v){
    R=r;
    V=v;
    cartesian_to_OE();
    set_TimePeriod();
}

//print the values of cartesian coordinates km, km/s
void Orbit::print_cartesian(){
    std::cout << "Radius: " << R.transpose() << std::endl;
    std::cout << "Velocity: " << V.transpose() << std::endl;
}
//print the values of perifocal coordinates km, km/s
void Orbit::print_perifocal(){
    std::cout << "Radius (perifocal): " << Rp.transpose() << std::endl;
    std::cout << "Velocity (perifocal): " << Vp.transpose() << std::endl;
}
//print the rotation matrices between perifocal and ECI frames
void Orbit::print_rotMat(){
    std::cout << "Perifocal to Cartesian \n" << Rptoc << std::endl;
    std::cout << "Cartesian to Perifocal \n" << Rctop << std::endl;
}
//convert Orbital elements to cartesian
void Orbit::OE_to_cartesian(){
    double p, r;

    p=a*(1-e*e);
    r=p/(1+e*cos(nu));

    //defining perifocal coordinates
    Rp(0)=r*cos(nu);
    Rp(1)=r*sin(nu);
    Rp(2)=0;

    Vp(0)=sqrt(mu/p)*(-sin(nu));
    Vp(1)=sqrt(mu/p)*(e+cos(nu));
    Vp(2)=0;

    //defining the rotation matrix
    Eigen::Matrix3d R3Omega;
    Eigen::Matrix3d R1i;
    Eigen::Matrix3d R3w;

    R3Omega <<  cos(RAAN), sin(RAAN), 0,
                -sin(RAAN), cos(RAAN), 0,
                0,0,1;
    
    R1i <<  1,0,0,
            0, cos(i), sin(i),
            0, -sin(i), cos(i);
    
    R3w <<  cos(AoP), sin(AoP), 0,
            -sin(AoP), cos(AoP), 0,
            0,0,1;
    
    Rctop = R1i*R3Omega;
    Rctop = R3w*Rctop;
    Rptoc = Rctop.transpose();
    //finding the cartesian coordianates

    R=Rptoc*Rp;
    V=Rptoc*Vp;
}
//convert cartesian to orbital elements
void Orbit::cartesian_to_OE(){
    Eigen::Vector3d H;
    Eigen::Vector3d N;
    Eigen::Vector3d E;

    double v, r;

    //calculating H vector and N vector
    H= R.cross(V);
    N << -H[1],H[0],0; //k X H

    // norm of V and R
    r=R.norm();
    v=V.norm();

    //Calculate a
    a= -mu/(2*((v*v/2)- (mu/r) ));

    //calculate E vector and e
    E=1/mu * ( (v*v - mu/r)*R - (R.dot(V))*V);
    e= E.norm();

    //calculate i
    i=acos(H[2]/H.norm());

    //calculate RAAN
    RAAN = acos(N[0]/N.norm());
    if(N[1]<0){
        RAAN=2*M_PI-RAAN;
    }
    //calculate AoP
    AoP=acos(N.dot(E)/(N.norm()*e));
    if (E[2]<0){
        AoP=2*M_PI-AoP;
    }
    //calculate nu
    nu=acos(E.dot(R)/(e*r));
    if (R.dot(V)<0){
        nu=2*M_PI-nu;
    }
}

// function for equation of motion for 2BP
Eigen::VectorXd Orbit::EoM_2BP(Eigen::Vector3d r, Eigen::Vector3d v){
    Eigen::VectorXd acc(6);
    acc(0)=v(0);
    acc(1)=v(1);
    acc(2)=v(2);
    acc(3)=-mu*r(0)/(r.norm()*r.norm()*r.norm());
    acc(4)=-mu*r(1)/(r.norm()*r.norm()*r.norm());
    acc(5)=-mu*r(2)/(r.norm()*r.norm()*r.norm());
    return acc;
}
// function for equation of motion for 2BP with thrust in velocity direction
Eigen::VectorXd Orbit::EoM_2BP_thrust_V(Eigen::Vector3d r, Eigen::Vector3d v){
    Eigen::VectorXd acc(6);
    acc(0)=v(0);
    acc(1)=v(1);
    acc(2)=v(2);
    acc(3)=-mu*r(0)/(r.norm()*r.norm()*r.norm()) + thrust*v(0)/(v.norm());
    acc(4)=-mu*r(1)/(r.norm()*r.norm()*r.norm()) + thrust*v(1)/(v.norm());
    acc(5)=-mu*r(2)/(r.norm()*r.norm()*r.norm()) + thrust*v(2)/(v.norm());
    return acc;
}

// function for equation of motion for 2BP with J2 perterbations
Eigen::VectorXd Orbit::EoM_2BP_J2(Eigen::Vector3d R, Eigen::Vector3d V){
    Eigen::VectorXd acc(6);
    double r;
    r=R.norm();
    acc(0)=V(0);
    acc(1)=V(1);
    acc(2)=V(2);
    acc(3)=-(mu*R(0)/(r*r*r))*(1 + (J2_E*3*RE*RE/(2*r*r)) - (J2_E*15*RE*RE*R(2)*R(2)/(2*std::pow(r,4))));
    acc(4)=-(mu*R(1)/(r*r*r))*(1 + (J2_E*3*RE*RE/(2*r*r)) - (J2_E*15*RE*RE*R(2)*R(2)/(2*std::pow(r,4))));
    acc(5)=-(mu*R(2)/(r*r*r))*(1 + (J2_E*9*RE*RE/(2*r*r)) - (J2_E*15*RE*RE*R(2)*R(2)/(2*std::pow(r,4))));
    return acc;
}

// Equations of Motion Function
Eigen::VectorXd Orbit::EoM(Eigen::Vector3d r, Eigen::Vector3d v, int EOM_int){
    if (EOM_int==0){
        return EoM_2BP(r,v);
    }
    if (EOM_int==1){
        return EoM_2BP_thrust_V(r,v);
    }
    if (EOM_int==2){
        return EoM_2BP_J2(r,v);
    }
    else{
        return EoM_2BP(r,v);
    }
    
}

//Private function to covert any cartesian coordinates to Orbital Elements
Eigen::ArrayXXd Orbit::cartesian_to_OE_I(Eigen::Vector3d x, Eigen::Vector3d y){
    Eigen::Vector3d H;
    Eigen::Vector3d N;
    Eigen::Vector3d E;

    double v, r;
    double a1, e1, i1, RAAN1, AoP1, nu1;

    //calculating H vector and N vector
    H= x.cross(y);
    N<<-H[1],H[0],0; //k X H

    // norm of V and R
    r=x.norm();
    v=y.norm();

    //Calculate a
    a1= -mu/(2*((v*v/2)- (mu/r) ));

    //calculate E vector and e
    E=1/mu * ( (v*v - mu/r)*x - (x.dot(y))*y);
    e1= E.norm();

    //calculate i
    i1=acos(H[2]/H.norm());

    //calculate RAAN
    RAAN1 = acos(N[0]/N.norm());
    if(N[1]<0){
        RAAN1=2*M_PI-RAAN1;
    }
    //calculate AoP
    AoP1=acos(N.dot(E)/(N.norm()*e1));
    if (E[2]<0){
        AoP1=2*M_PI-AoP1;
    }
    //calculate nu
    nu1=acos(E.dot(x)/(e1*r));
    if (x.dot(y)<0){
        nu1=2*M_PI-nu1;
    }
    Eigen::ArrayXXd OE1(6,1);
    OE1(0,0)=a1;
    OE1(1,0)=e1;
    OE1(2,0)=i1;
    OE1(3,0)=RAAN1;
    OE1(4,0)=AoP1;
    OE1(5,0)=nu1;
    return OE1;
}

Eigen::Vector3d Orbit::angular_mom(Eigen::Vector3d x, Eigen::Vector3d y){
    Eigen::Vector3d H;
    //calculating H vector and N vector
    H= x.cross(y);
    return H;
}

double Orbit::orbital_energy(Eigen::Vector3d x, Eigen::Vector3d y, int EOM_int){
    double orbital_energy, r,v;
    r=x.norm();
    v=y.norm();
    orbital_energy= v*v/2-mu/r;
    if (EOM_int==0){// ideal orbit
        orbital_energy=orbital_energy;
    }
    if (EOM_int==1){//thrust in V direction
        orbital_energy=orbital_energy;
    }
    if (EOM_int==2){//J2
        orbital_energy=orbital_energy+(mu*J2_E*RE*RE/(2*r*r*r))*(3*x(2)*x(2)/(r*r)-1);
    }

    return orbital_energy;
}

void Orbit::propagate_2BP(double step,double trange, int EOM_int, std::string name){
    int t=0;
    //number of steps
    int noi=trange/step;

    Eigen::ArrayXXd R_propagated(3,noi+1);
    Eigen::ArrayXXd V_propagated(3,noi+1);
    Eigen::ArrayXXd Acc_propagated(3,noi+1);
    Eigen::ArrayXXd H_propagated(3,noi+1);

    Eigen::ArrayXXd energy_propagated(1,noi+1);

    Eigen::ArrayXXd a_propagated(1,noi+1);
    Eigen::ArrayXXd e_propagated(1,noi+1);
    Eigen::ArrayXXd i_propagated(1,noi+1);
    Eigen::ArrayXXd RAAN_propagated(1,noi+1);
    Eigen::ArrayXXd AoP_propagated(1,noi+1);
    Eigen::ArrayXXd nu_propagated(1,noi+1);

    Eigen::ArrayXXd time(1,noi+1);

    Eigen::ArrayXXd y(6,1);
    Eigen::ArrayXXd k1(6,1);
    Eigen::ArrayXXd k2(6,1);
    Eigen::ArrayXXd k3(6,1);
    Eigen::ArrayXXd k4(6,1);

    // setting initial values
    R_propagated.col(0)=R;
    V_propagated.col(0)=V;
    Acc_propagated.col(0)=-mu/(std::pow(R.norm(),3))*R;
    H_propagated.col(0)=angular_mom(R,V);
    energy_propagated.col(0)=orbital_energy(R,V,EOM_int);
    a_propagated.col(0)=a;
    e_propagated.col(0)=e;
    i_propagated.col(0)=i;
    RAAN_propagated.col(0)=RAAN;
    AoP_propagated.col(0)=AoP;
    nu_propagated.col(0)=nu;
    time.col(0)=0;
    y.block(0,0,3,1)=R;
    y.block(3,0,3,1)=V;
    while (t<noi){
        //RK4 Integrator
        k1= step*EoM(R_propagated.col(t),V_propagated.col(t), EOM_int);
        k2= step*EoM(R_propagated.col(t) + k1.block(0,0,3,1)/2,V_propagated.col(t)+ k1.block(3,0,3,1)/2, EOM_int);
        k3= step*EoM(R_propagated.col(t) + k2.block(0,0,3,1)/2,V_propagated.col(t)+ k2.block(3,0,3,1)/2, EOM_int);
        k4= step*EoM(R_propagated.col(t) + k3.block(0,0,3,1),V_propagated.col(t)+ k3.block(3,0,3,1), EOM_int);
        y += (k1+2*k2+2*k3+k4)/6;

        //Increasing Time Step
        t++;
        //set time
        time.col(t)=step*t;
        //set R, V and Acc
        R_propagated.col(t)=y.block(0,0,3,1);
        V_propagated.col(t)=y.block(3,0,3,1);
        Eigen::Vector3d rtest;
        rtest=R_propagated.col(t);
        Acc_propagated.col(t)=-mu/(std::pow(rtest.norm(),3))*R;
        //set angular momentum and orbital energy
        H_propagated.col(t)=angular_mom(R_propagated.col(t),V_propagated.col(t));
        energy_propagated.col(t)=orbital_energy(R_propagated.col(t),V_propagated.col(t), EOM_int);
        //convert to Orbital Elements
        Eigen::ArrayXXd OrbitalElmts(6,1);
        OrbitalElmts=cartesian_to_OE_I(R_propagated.col(t),V_propagated.col(t));
        //set orbital elements
        a_propagated.col(t)=OrbitalElmts(0);
        e_propagated.col(t)=OrbitalElmts(1);
        i_propagated.col(t)=OrbitalElmts(2);
        RAAN_propagated.col(t)=OrbitalElmts(3);
        AoP_propagated.col(t)=OrbitalElmts(4);
        nu_propagated.col(t)=OrbitalElmts(5);
    }
    // Eigen Matrix print format to write csv files
    Eigen::IOFormat csv(10, 0, ", ", "\n", "", "", "", "");
    // Write file with R vector;
    // Matrix of all data
    Eigen::ArrayXXd Matrice(noi+1,20);
    //time
    Matrice.col(0)=time.transpose();
    //orbital elements
    Matrice.col(1)=a_propagated.transpose();
    Matrice.col(2)=e_propagated.transpose();
    Matrice.col(3)=i_propagated.transpose()*180.0/M_PI;
    Matrice.col(4)=RAAN_propagated.transpose()*180.0/M_PI;
    Matrice.col(5)=AoP_propagated.transpose()*180.0/M_PI;
    Matrice.col(6)=nu_propagated.transpose()*180.0/M_PI;
    //energy
    Matrice.col(7)=energy_propagated.transpose();
    //Radius
    Matrice.col(8)=R_propagated.row(0);
    Matrice.col(9)=R_propagated.row(1);
    Matrice.col(10)=R_propagated.row(2);
    //Velocity
    Matrice.col(11)=V_propagated.row(0);
    Matrice.col(12)=V_propagated.row(1);
    Matrice.col(13)=V_propagated.row(2);
    //Acceleration
    Matrice.col(14)=Acc_propagated.row(0);
    Matrice.col(15)=Acc_propagated.row(1);
    Matrice.col(16)=Acc_propagated.row(2);
    //Angular Momentum
    Matrice.col(17)=H_propagated.row(0);
    Matrice.col(18)=H_propagated.row(1);
    Matrice.col(19)=H_propagated.row(2);
    std::ofstream theFile;
    theFile.open("Results/" + name + "_file.csv");
    theFile << "Time (sec),Semi-Major Axis (km),Eccentricity,Inclination (deg),RAAN (deg),Argument of Periapsis (deg),True Anomaly (deg),Orbital Energy (km^2/sec^2),Radius_1 (km),Radius_2 (km),Radius_3 (km),Velocity_1 (km/s),Velocity_2 (km/s),Velocity_3 (km/s),Acceleration_1 (km/s^2),Acceleration_2 (km/s^2),Acceleration_3 (km/s^2),Angular_Momemntum_1 (km^2/s),Angular_Momemntum_2 (km^2/s),Angular_Momemntum_3 (km^2/s)" << std::endl;
    theFile << Matrice.format(csv)<< std::endl;
    theFile.close();
}


// Kepler's Equation Solver
 Eigen::VectorXd Orbit::Kepler_prob(Eigen::Vector3d R0, Eigen::Vector3d V0, double del_t){
    double alpha=1/a;
    double chi_0;
    double chi=chi_0;
    double z, t;
    double tol=1e-8;
    double ratio=1;
    double r0=R0.norm();
    double v0=V0.norm();
    if(e>1){
        double x=-2*mu*del_t/(a*(R0.dot(V0)+del_t*std::sqrt(-a*mu)*(1-r0/a)));
        chi_0=del_t*std::sqrt(-a)*std::log(x);
    }
    else {
        chi_0= std::sqrt(mu) * std::abs(alpha) *del_t;
    }
    int i=0;
    int maxiter=10000;
    while(std::abs(ratio)>tol){
        if(i>maxiter){
            break;}
        z=chi*chi*alpha;
        double f_chi, f_dash_chi;
        f_chi=R0.dot(V0)*chi*chi*stumpff_C(z)/(std::sqrt(mu));
        f_chi=f_chi+r0*chi*(1-z*stumpff_S(z));
        f_chi=f_chi+ std::pow(chi,3)*stumpff_S(z)- std::sqrt(mu)*del_t;

        f_dash_chi=R0.dot(V0)*chi*(1-z*stumpff_S(z))/(std::sqrt(mu));
        f_dash_chi=f_dash_chi + chi*chi*stumpff_C(z) + r0 -z*r0*stumpff_C(z);

        ratio=f_chi/f_dash_chi;
        chi=chi-ratio;
        i++;
    }
    
    z=chi*chi*alpha;
    double f, g, f_dot, g_dot;
    Eigen::Vector3d R_t, V_t;
    f= 1- chi*chi*stumpff_C(z)/r0;
    g=del_t - (chi*chi*chi*stumpff_S(z))/std::sqrt(mu);

    R_t=f*R0 +g*V0;
    double r_t=R_t.norm();
    f_dot=(std::sqrt(mu)/(r0*r_t)) * chi*(z*stumpff_S(z)-1);
    g_dot=1- chi*chi*stumpff_C(z)/r_t;

    V_t=f_dot*R0 + g_dot*V0;
    Eigen::VectorXd res(6);
    res.block(0,0,3,1)=R_t;
    res.block(3,0,3,1)=V_t;
    return res;
 }