#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <random>
#include <vector>
#include <eigen-5.0.0/Eigen/Dense>

class MonteCarlo {
public:
    MonteCarlo();
    void run_simulation(int num_trials, std::string name);

    void define_gmms(std::string param, const std::vector<std::tuple<double, double, double>>& components);

    void define_eccentricity_params(double mu, double std);

private:

    // RNG for sampling
    std::mt19937 gen;

    // Gaussian mixture model component
    struct GaussianComponent {
        double weight; // w_k (must sum to 1.0 across all components)
        double mean;   // μ_k
        double stdDev; // σ_k
    };

    // Defined GMM parameters for each element (stored by value)
    std::vector<GaussianComponent> i_gmm;
    std::vector<GaussianComponent> a_gmm;
    
    // Parameters for eccentricity (log-normal)
    double e_mu;
    double e_std;

    // samples a random variable from a Gaussian Mixture Model (GMM)
    double sample_gmm(const std::vector<GaussianComponent>& gmm, std::mt19937& gen);

    // sample a orbit
    Eigen::VectorXd sample_orbit();
    
    int successful_detections = 0;
    int total_trials = 0;
};

#endif // MONTE_CARLO_H