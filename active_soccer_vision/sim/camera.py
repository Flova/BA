import math
import cv2
import numpy as np
import transforms3d

class Camera:
    def __init__(self, fov=45, height=1080, width=1920):
        self.fov = fov
        self.height = height
        self.width = width

        T_cam = [4.5, 3.0, 1.0]
        R_cam = transforms3d.euler.euler2mat(math.pi/2, math.radians(10), 0)
        self.camera_frame = transforms3d.affines.compose(T_cam, R_cam, np.ones(3))

        R_camera_optical_frame = [[0,0,1], [0, -1, 0], [1, 0, 0]]
        self.camera_optical_frame = transforms3d.affines.compose(np.zeros(3), R_camera_optical_frame, np.ones(3))


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

    def get_point_in_camera_optical_frame(self, point_in_world_frame):
        point = self.get_point_in_camera_frame(point_in_world_frame)

        return np.matmul(np.linalg.inv(self.camera_optical_frame), point)

    def check_if_point_is_visible(self, point):

        if point.shape[0] == 2:
            point = np.append(point, 0)

        R_point_on_map = transforms3d.euler.euler2mat(0.0, 0, 0)
        A_point_on_map = transforms3d.affines.compose(point, R_point_on_map, np.ones(3))

        A_ball_in_cam_optical_frame = self.get_point_in_camera_optical_frame(A_point_on_map)

        p, _, _, _ = transforms3d.affines.decompose(A_ball_in_cam_optical_frame)
        
        if p[2] >= 0:
            p_pixel = np.matmul(self.K, p)
            p_pixel = p_pixel * (1/p_pixel[2])
            
            if 0 < p_pixel[0] <= self.width and 0 < p_pixel[1] <= self.height:
                return True
        return False

    def get_pixel_position_in_world(self, point): 
        A_field_normal = transforms3d.affines.compose([0, 0, 1], np.eye(3), np.ones(3))
        A_field_normal = self.get_point_in_camera_optical_frame(A_field_normal)

        A_field_point = transforms3d.affines.compose(np.zeros(3), np.eye(3), np.ones(3))
        A_field_point = self.get_point_in_camera_optical_frame(A_field_point)

        field_normal, _, _ , _ = transforms3d.affines.decompose(A_field_normal)
        field_point , _, _ , _ = transforms3d.affines.decompose(A_field_point)

        field_normal = field_point - field_normal

        point = self._get_field_intersection_for_pixels(np.array([[point[0], point[1], 0]]), (field_normal, field_point))[0]

        point = transforms3d.affines.compose(point, np.eye(3), np.ones(3))

        point = np.matmul(np.matmul(self.camera_frame, self.camera_optical_frame), point)

        point, _, _ , _ = transforms3d.affines.decompose(point)

        return point
        
        
    def _get_field_intersection_for_pixels(self, points, field):
        """
        Projects an numpy array of points to the correspoding places on the field plane (in the camera frame).
        """
        camera_projection_matrix = self.K.reshape(-1)

        points[:, 0] = (points[:, 0] - (camera_projection_matrix[2])) / camera_projection_matrix[0]
        points[:, 1] = (points[:, 1] - (camera_projection_matrix[5])) / camera_projection_matrix[4]
        points[:, 2] = 1

        intersections = self._line_plane_intersections(field[0], field[1], points)

        return intersections

    def _line_plane_intersections(self, plane_normal, plane_point, ray_directions):
        n_dot_u = np.tensordot(plane_normal, ray_directions, axes=([0],[1]))
        relative_ray_distance = -plane_normal.dot(- plane_point) / n_dot_u

        # we are casting a ray, intersections need to be in front of the camera
        relative_ray_distance[relative_ray_distance <= 0] = 1000

        ray_directions[:,0] = np.multiply(relative_ray_distance, ray_directions[:,0])
        ray_directions[:,1] = np.multiply(relative_ray_distance, ray_directions[:,1])
        ray_directions[:,2] = np.multiply(relative_ray_distance, ray_directions[:,2])

        return ray_directions

    def get_projected_image_corners(self):
        return np.array([
            self.get_pixel_position_in_world(np.array([self.width, 0, 1.0]))[:2],
            self.get_pixel_position_in_world(np.array([self.width, self.height, 1.0]))[:2],
            self.get_pixel_position_in_world(np.array([0, self.height, 1.0]))[:2],
            self.get_pixel_position_in_world(np.array([0, 0, 1.0]))[:2]])


if __name__ == '__main__':
    print("Testing point visibility test")

    point = np.array([5.5, 3.0])

    cam = Camera()

    print(cam.check_if_point_is_visible(point))



