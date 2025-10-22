import numpy as np
from localization.random_numbers import rand_von_mises, randn
from scipy.stats import norm


class Particle(object):
    """Data structure for storing particle information (state and weight)"""
    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=0.0):
        self.x = x
        self.y = y
        self.theta = np.mod(theta, 2.0*np.pi)
        self.weight = weight

    def getX(self):
        return self.x
        
    def getY(self):
        return self.y
        
    def getTheta(self):
        return self.theta
        
    def getWeight(self):
        return self.weight

    def setX(self, val):
        self.x = val

    def setY(self, val):
        self.y = val

    def setTheta(self, val):
        self.theta = np.mod(val, 2.0*np.pi)

    def setWeight(self, val):
        self.weight = val


def estimate_pose(particles_list):
    """Estimate the pose from particles by computing the average position and orientation over all particles. 
    This is not done using the particle weights, but just the sample distribution."""
    x_sum = 0.0
    y_sum = 0.0
    cos_sum = 0.0
    sin_sum = 0.0
     
    for particle in particles_list:
        x_sum += particle.getX()
        y_sum += particle.getY()
        cos_sum += np.cos(particle.getTheta())
        sin_sum += np.sin(particle.getTheta())
        
    flen = len(particles_list)
    if flen != 0:
        x = x_sum / flen
        y = y_sum / flen
        theta = np.arctan2(sin_sum/flen, cos_sum/flen)
    else:
        x = x_sum
        y = y_sum
        theta = 0.0
        
    return Particle(x, y, theta)
     
     
def move_particle(particle, delta_x, delta_y, delta_theta):
    particle.x += + delta_x
    particle.y += delta_y
    new_theta = particle.getTheta() + delta_theta

    particle.theta = np.mod(new_theta, 2.0 * np.pi)


def add_uncertainty(particles_list, sigma, sigma_theta):
    """Add some noise to each particle in the list. Sigma and sigma_theta is the noise
    variances for position and angle noise."""
    for particle in particles_list:
        particle.x += randn(0.0, sigma)
        particle.y += randn(0.0, sigma)
        particle.theta = np.mod(particle.theta + randn(0.0, sigma_theta), 2.0 * np.pi) 


def add_uncertainty_von_mises(particles_list, sigma, theta_kappa):
    """Add some noise to each particle in the list. Sigma and theta_kappa is the noise
    variances for position and angle noise."""
    for particle in particles_list:
        particle.x += randn(0.0, sigma)
        particle.y += randn(0.0, sigma)
        particle.theta = np.mod(rand_von_mises(particle.theta, theta_kappa), 2.0 * np.pi) - np.pi


def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        p = Particle(520.0*np.random.ranf() - 120.0, 420.0*np.random.ranf() - 120.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

def sample_motion_model(particles_list, distance, angle, sigma_d, sigma_theta):
    for p in particles_list:
        delta_x = distance * np.cos(p.getTheta() + angle)
        delta_y = distance * np.sin(p.getTheta() + angle)
    
        move_particle(p, delta_x, delta_y, angle)
    if not(distance == 0 and angle == 0):
        add_uncertainty(particles_list, sigma_d, sigma_theta)
        sigma_d = sigma_d if distance != 0 else sigma_d * 0.1
        add_uncertainty(particles_list, sigma_d, sigma_theta)


def measurement_model(particle_list, ObjectIDs, landmarkIDs, landmarks, dists, angles, sigma_d, sigma_theta):
    for particle in particle_list:
        x_i = particle.getX()
        y_i = particle.getY()
        theta_i = particle.getTheta()

        p_observation_given_x = 1.0

        #p(z|x) = sum over the probability for all landmarks
        for landmarkID, dist, angle in zip(ObjectIDs, dists, angles):
            if landmarkID in landmarkIDs:
                l_x, l_y = landmarks[landmarkID]
                d_i = np.sqrt((l_x - x_i)**2 + (l_y - y_i)**2)

                p_d_m = norm.pdf(dist, loc=d_i, scale=sigma_d)

                e_theta = np.array([np.cos(theta_i), np.sin(theta_i)])
                e_theta_hat = np.array([-np.sin(theta_i), np.cos(theta_i)])

                e_l = np.array([l_x - x_i, l_y - y_i]) / d_i

                dot = np.clip(np.dot(e_l, e_theta), -1.0, 1.0)
                phi_i = np.sign(np.dot(e_l, e_theta_hat)) * np.arccos(dot)
                
                p_phi_m = norm.pdf(angle,loc=phi_i, scale=sigma_theta)


                p_observation_given_x *= (p_d_m * p_phi_m)

        particle.setWeight(p_observation_given_x)

def inject_random_particles(particle_list, ratio=0.01):
    n_random = int(len(particle_list) * ratio)
    for i in range(n_random):
        particle_list[i] = Particle(
            520.0 * np.random.ranf() - 120.0,
            420.0 * np.random.ranf() - 120.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / len(particle_list)
        )
    return particle_list

def resample_particles(particle_list):
    weights = np.array([p.getWeight() for p in particle_list])
    total_weight = np.sum(weights)
    weights /= total_weight

    cdf = np.cumsum(weights)
    resampled = []

    for _ in range(len(particle_list)):
        z = np.random.rand()
        idx = np.searchsorted(cdf, z)
        p_resampled = Particle(
            particle_list[idx].getX(),
            particle_list[idx].getY(),
            particle_list[idx].getTheta(),
            1.0 / len(particle_list)
        )
        resampled.append(p_resampled)

    return resampled
