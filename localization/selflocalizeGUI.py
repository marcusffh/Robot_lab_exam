import cv2
import numpy as np
import sys
import matplotlib.cm as cm

class SelflocalizeGUI(object):
    def __init__(self, landmarkIDs, landmark_colors, landmarks):
        self.landmarkIDs = landmarkIDs
        self.landmark_colors = landmark_colors
        self.landmarks = landmarks
        self.CMAGENTA = (255, 0, 255)
        self.CWHITE = (255, 255, 255)

    def jet(self, x):
        c = cm.jet(x)  # returns RGBA in [0,1]
        return (int(c[2]*255), int(c[1]*255), int(c[0]*255))  # convert to BGR for OpenCV


    def draw_world(self, est_pose, particles, world):
        offsetX = 100
        offsetY = 250
        ymax = world.shape[0]
        world[:] = self.CWHITE

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

        for i, ID in enumerate(self.landmarkIDs):
            lm_x, lm_y = self.landmarks[ID]
            lm_screen = (int(lm_x + offsetX), int(ymax - (lm_y + offsetY)))
            cv2.circle(world, lm_screen, 5, self.landmark_colors[i], -1)

        a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
        b = (
            int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
            ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
        )
        cv2.circle(world, a, 5, self.CMAGENTA, 2)
        cv2.line(world, a, b, self.CMAGENTA, 2)
