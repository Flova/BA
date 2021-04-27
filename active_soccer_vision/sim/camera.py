import math
import cv2
import numpy as np
import transforms3d

class Camera:
    def __init__(self, fov=45, height=1080, width=1920):
        self.fov = fov
        self.height = height
        self.width = width

        T_cam = [5.0, 5.0, 1.0]
        R_cam = transforms3d.euler.euler2mat(0, math.radians(45), 0)
        self.camera_frame = transforms3d.affines.compose(T_cam, R_cam, np.ones(3))

        def mat_from_fov_and_resolution(fov, res):
            return 0.5 * res * (math.cos((fov / 2)) / math.sin((fov / 2)))

        def h_fov_to_v_fov(h_fov, height, width):
            return 2 * math.atan(math.tan(h_fov * 0.5) * (height / width))

        f_y = mat_from_fov_and_resolution(
                h_fov_to_v_fov(self.fov, self.height, self.width),
                self.height)
        f_x = mat_from_fov_and_resolution(self.fov, self.width)

        self.K = np.array(
            [f_x, 0, self.width / 2,
            0, f_y, self.height / 2,
            0, 0, 1]).reshape((3,3))

    def get_point_in_camera_frame(self, point_in_world_frame):
        return np.matmul(np.linalg.inv(self.camera_frame), point_in_world_frame)

    def check_if_point_is_visible(self, point):
        A_ball_in_cam_frame = self.get_point_in_camera_frame(point)

        T_camera_optical_frame = [1.0, 0, 0]
        R_camera_optical_frame = [[0,0,1], [0, -1, 0], [1, 0, 0]]
        A_camera_optical_frame = transforms3d.affines.compose(T_camera_optical_frame, R_camera_optical_frame, np.ones(3))

        A_ball_in_cam_optical_frame = np.matmul(np.linalg.inv(A_camera_optical_frame), A_ball_in_cam_frame)

        p, _, _, _ = transforms3d.affines.decompose(A_ball_in_cam_optical_frame)
        
        if p[2] >= 0:
            p_pixel = np.matmul(self.K, p)
            p_pixel = p_pixel * (1/p_pixel[2])
            
            if 0 < p_pixel[0] <= self.width and 0 < p_pixel[1] <= self.height:
                return True
        return False


if __name__ == '__main__':
    print("Testing point visibility test")

    T_ball_on_map = [1.0, 0, 0]
    R_ball_on_map = transforms3d.euler.euler2mat(0.0, 0, 0)
    A_ball_on_map = transforms3d.affines.compose(T_ball_on_map, R_ball_on_map, np.ones(3))

    cam = Camera()

    print(cam.check_if_point_is_visible(A_ball_on_map))