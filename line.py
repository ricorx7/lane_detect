import numpy as np
import cv2


class Line:
    """
    A Line is defined from two points (x1, y1) and (x2, y2) as follows:
    y - y1 = (y2 - y1) / (x2 - x1) * (x - x1)
    Each line has its own slope and intercept (bias).
    """
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, img, color=[255, 0, 0], thickness=10, offset_x=0, offset_y=0):
        #cv2.line(img, (self.x1+offset_x, self.y1+offset_y), (self.x2+offset_x, self.y2+offset_y), color, thickness)
        #print("X1: ", self.x1)
        #print("Y1: ", self.y1)
        #print("X2: ", self.x2)
        #print("Y2: ", self.y2)
        x1 = int(self.x1)
        y1 = int(self.y1)
        x2 = int(self.x2)
        y2 = int(self.y2)
        #print("X1_: ", x1)
        #print("Y1_: ", y1)
        #print("X2_: ", x2)
        #print("Y2_: ", y2)
        cv2.line(img, (x1+offset_x, y1+offset_y), (x2+offset_x, y2+offset_y), color, thickness)
