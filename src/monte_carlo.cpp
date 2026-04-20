#include "monte_carlo.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

MonteCarlo::MonteCarlo(){
    // Seed RNG with random device
    std::random_device rd;
    gen = std::mt19937(rd());
};


// define GMMs for parameters
void MonteCarlo::define_gmms(std::string param, const std::vector<std::tuple<double, double, double>>& components) {
    std::vector<GaussianComponent> gmm;
    for (const auto& comp : components) {
        GaussianComponent gc;
        gc.weight = std::get<0>(comp);
        gc.mean = std::get<1>(comp);
        gc.stdDev = std::get<2>(comp);
        gmm.push_back(gc);
    }

    if (param == "i") {
        i_gmm = gmm;
    } else if (param == "a") {
        a_gmm = gmm;
    } else {
        // Handle unknown parameter
        throw std::invalid_argument("Unknown parameter for GMM definition: " + param);
    }
}

// define eccentricity parameters
void MonteCarlo::define_eccentricity_params(double mu, double std) {
    e_mu = mu;
    e_std = std;
}

// samples a random variable from a Gaussian Mixture Model (GMM)
double MonteCarlo::sample_gmm(const std::vector<GaussianComponent>& mixture, std::mt19937& gen) {
    

    // Define weights for the selection
    std::vector<double> weights;
    for (const auto& comp : mixture) {
        weights.push_back(comp.weight);
    }

    // Use a discrete distribution to select an index (0, 1, 2, ...) based on the weights
    std::discrete_distribution<> component_selector(weights.begin(), weights.end());
    int selected_index = component_selector(gen);

    // Get the parameters of the selected component
    const auto& selected_comp = mixture[selected_index];


    // Use the standard C++ normal distribution
    std::normal_distribution<> normal_dist(selected_comp.mean, selected_comp.stdDev);
    
    // Draw the sample and return
    return normal_dist(gen);
}

// sample orbit using distributions
Eigen::VectorXd MonteCarlo::sample_orbit() {
    // Sample inclination from GMM
    double sampled_i = sample_gmm(i_gmm, gen);
    // Sample semi-major axis from GMM
    double sampled_a = sample_gmm(a_gmm, gen);
    // Sample eccentricity from log-normal distribution
    std::lognormal_distribution<> lognormal_dist(e_mu, e_std);
    double sampled_e = lognormal_dist(gen);
    // Sample other elements from uniform distributions
    double sampled_RAAN = std::uniform_real_distribution<>(0, 360)(gen);
    double sampled_AoP = std::uniform_real_distribution<>(0, 360)(gen);
    double sampled_nu = std::uniform_real_distribution<>(0, 360)(gen);

    Eigen::VectorXd orbit_elements(6);
    orbit_elements << sampled_a, sampled_e, sampled_i, sampled_RAAN, sampled_AoP, sampled_nu;
    return orbit_elements;
}

// run simulation
void MonteCarlo::run_simulation(int num_trials, std::string name) {
    successful_detections = 0;
    total_trials = num_trials;

    Eigen::ArrayXXd sampled_orbits(7, num_trials);

    for (int trial = 0; trial < num_trials; ++trial) {
        Eigen::VectorXd orbit = sample_orbit();
        Eigen::VectorXd orbit_with_trial(7);
        orbit_with_trial << trial, orbit;
        sampled_orbits.col(trial) = orbit_with_trial;
    }

    // save sampled orbits as csv
    Eigen::IOFormat csv(10, 0, ", ", "\n", "", "", "", "");
    std::ofstream theFile;
    theFile.open("Results/"+ name + "_sampled_orbits.csv");
    if (theFile.is_open()){
        theFile << "trial #, a (km), e, i (deg), RAAN (deg), AoP (deg), nu (deg)\n";
        theFile << sampled_orbits.transpose().format(csv) << std::endl;
        theFile.close();
    }

    // Output results
    std::cout << "Total Trials: " << total_trials << std::endl;
    std::cout << "Successful Detections: " << successful_detections << std::endl;
    std::cout << "Detection Rate: " << (static_cast<double>(successful_detections) / total_trials) * 100.0 << "%" << std::endl;
}