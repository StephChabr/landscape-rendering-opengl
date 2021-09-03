from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians
from utils import rand_cam2, rand_cam, rand_cam3
import numpy as np

class Camera:
    def __init__(self):
        self.camera_pos = Vector3([0, 2000., 0])
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        self.camera_right = Vector3([1.0, 0.0, 0.0])

        self.mouse_sensitivity = 0.25
        self.jaw = 220
        self.pitch = 0

        self.update_camera_vectors()

    def get_params(self):
        return [self.camera_pos[2],self.camera_pos[0],self.camera_pos[1],self.jaw,self.pitch]

    def random_cam(self):
        [cy,cx,cz],cyaw,cpitch = rand_cam3()
        self.camera_pos = Vector3([cx,cz,cy])
        self.jaw = cyaw
        self.pitch = cpitch
        self.update_camera_vectors()
    
    def set_cam(self,params):
        [cy,cx,cz,cyaw,cpitch] = params.tolist()
        self.camera_pos = Vector3([cx,cz,cy])
        self.jaw = cyaw
        self.pitch = cpitch
        self.update_camera_vectors()

    def get_view_matrix(self):
        return matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.jaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89:
                self.pitch = 89
            if self.pitch < -89:
                self.pitch = -89

        self.update_camera_vectors()

    def update_camera_vectors(self):
        front = Vector3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.jaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.jaw)) * cos(radians(self.pitch))

        self.camera_front = vector.normalise(front)
        self.camera_right = vector.normalise(vector3.cross(self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalise(vector3.cross(self.camera_right, self.camera_front))

    # Camera method for the WASD movement
    def process_keyboard(self, direction, velocity):
        if direction == "FORWARD":
            self.camera_pos += self.camera_front * velocity
        if direction == "BACKWARD":
            self.camera_pos -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_pos -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_pos += self.camera_right * velocity