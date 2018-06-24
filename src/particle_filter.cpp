#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cmath>
#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std_devs[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.

	if (is_initialized) return;

	double std_x = std_devs[0];
	double std_y = std_devs[1];
	double std_theta = std_devs[2];

	// Create normal (Gaussian) distribution for x, y and theta.
	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	// Initialize the particles
	for (auto&& particle : particles) {
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 0.0;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_devs[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

	double std_x = std_devs[0];
	double std_y = std_devs[1];
	double std_theta = std_devs[2];

	for (auto&& particle : particles) {
		// Modify particle position based on velocity and yaw rate
		double x_diff = velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
		double y_diff = velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));

		particle.x += std::isnan(x_diff) ? 0.0 : x_diff;
		particle.y += std::isnan(y_diff) ? 0.0 : y_diff;
		particle.theta += delta_t * yaw_rate;


		// Add gaussian noise to each particles x, y and theta.
		std::normal_distribution<double> dist_x(particle.x, std_x);
		std::normal_distribution<double> dist_y(particle.y, std_y);
		std::normal_distribution<double> dist_theta(particle.theta, std_theta);
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto&& observation : observations) {
		double min_distance_squared = std::numeric_limits<double>::max();
		auto min_prediction_id  = -1;
		for (auto&& prediction : predicted) {
			double distance_squared = (observation.x - prediction.x) * (observation.x - prediction.x)
									+ (observation.y - prediction.y) * (observation.y - prediction.y);
			if (distance_squared < min_distance_squared) {
				min_distance_squared = distance_squared;
				min_prediction_id = prediction.id;
			}
		}
		observation.id = min_prediction_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	const double sensor_range_squared = sensor_range * sensor_range;
	const double sig_x = std_landmark[0];
	const double sig_y = std_landmark[1];

	for (auto&& particle : particles) {
		// Eliminate each map_landmark out of the particles sensor range
		std::vector<LandmarkObs> valid_map_landmarks(map_landmarks.landmark_list.size());
		for (auto&& landmark : map_landmarks.landmark_list) {
			const double distance_squared = (particle.x - landmark.x_f) * (particle.x - landmark.x_f)
										  + (particle.y - landmark.y_f) * (particle.y - landmark.y_f);
			if (distance_squared < sensor_range_squared) {
				valid_map_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}
		// Map the observations from vehicle coordinates(of the particle) to the global map coordinates
		std::vector<LandmarkObs> map_observations(observations);
		for (auto&& map_observation : map_observations) {
			double map_x = particle.x + cos(particle.theta) * map_observation.x - sin(particle.theta) * map_observation.y;
			double map_y = particle.y + sin(particle.theta) * map_observation.x + cos(particle.theta) * map_observation.y;
			map_observation.x = map_x;
			map_observation.y = map_y;
		}

		// Associate observations to landmarks
		dataAssociation(valid_map_landmarks, map_observations);

		// initialize particle weight for re-calculation
		particle.weight = 1.0;

		// calculate new weight
		for (auto&& valid_map_landmark : valid_map_landmarks) {
			for (auto&& map_observation : map_observations) {
				if (valid_map_landmark.id == map_observation.id) {
					// calculate normalization term
					double gauss_norm = (1.0/(2.0 * M_PI * sig_x * sig_y));
					// calculate exponent
					double exponent = ((map_observation.x - valid_map_landmark.x)*(map_observation.x - valid_map_landmark.x)) / (2 * sig_x * sig_x)
								    + ((map_observation.y - valid_map_landmark.y)*(map_observation.y - valid_map_landmark.y)) / (2 * sig_y * sig_y);
					// calculate weight using normalization terms and exponent
					double weight = gauss_norm * exp(-exponent);
					particle.weight *= weight;
				}
			}
		}
	}
}


void ParticleFilter::resample() {

	// Resample particles with replacement with probability proportional to their weight.

	// Update the copy of particle weights
	weights.clear();
	for (auto&& particle : particles) {
		weights.push_back(particle.weight);
	}
	// New distribution of indexes based on the particle weights
	std::discrete_distribution<unsigned> new_dist_index(weights.begin(),weights.end());

	// Resample particles
	std::vector<Particle> new_particles(num_particles);
	for (auto i=0; i < num_particles; i++) {
		unsigned picked_index = new_dist_index(gen);
		new_particles.push_back(particles[picked_index]);
	}
	particles.assign(new_particles.begin(), new_particles.end());
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

std::string ParticleFilter::getAssociations(Particle best) {
	std::vector<int> v = best.associations;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best) {
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best) {
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
