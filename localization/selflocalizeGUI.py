import cv2
import numpy as np
import random

class SelflocalizeGUI:

    def __init__(self, landmarks):
        """
        landmarks: list of Landmark objects
        """
        self.landmarks = landmarks
        self.CMAGENTA = (255, 0, 255)
        self.CWHITE = (255, 255, 255)

        # Assign each landmark a random color
        self.landmark_colors = {lm.id: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ) for lm in landmarks}

    def jet(self, x):
        r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
        g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
        b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)
        return (255.0*r, 255.0*g, 255.0*b)

    def draw_world(self, est_pose, particles, world):
        offsetX = 100
        offsetY = 250
        ymax = world.shape[0]
        world[:] = self.CWHITE

        # Draw particles
        max_weight = max([p.getWeight() for p in particles]) if particles else 1.0
        for particle in particles:
            x = int(particle.getX() + offsetX)
            y = ymax - int(particle.getY() + offsetY)
            colour = self.jet(particle.getWeight() / max_weight)
            cv2.circle(world, (x, y), 2, colour, 2)
            b = (
                int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
                ymax - int(particle.getY() + 15.0 * np.sin(particle.getTheta()) + offsetY),
            )
            cv2.line(world, (x, y), b, colour, 2)

        # Draw landmarks
        for lm in self.landmarks:
            lm_screen = (int(lm.x + offsetX), ymax - int(lm.y + offsetY))
            cv2.circle(world, lm_screen, int(lm.radius), self.landmark_colors[lm.id], -1)

        # Draw estimated robot pose
        a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
        b = (
            int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
            ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
        )
        cv2.circle(world, a, 5, self.CMAGENTA, 2)
        cv2.line(world, a, b, self.CMAGENTA, 2)
