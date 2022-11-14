import pyzed.sl as sl
import numpy as np
import torch


from functools import wraps
import time


def timeit(func):
    """Time function execution"""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} "
              f"Took {total_time:.4f} seconds")
        return result
    return timeit_wrapper


def get_gaze(obj):
    """ Create gaze defined from ears and nose """
    left_ear = obj.keypoint[sl.BODY_PARTS_POSE_18.LEFT_EAR.value]
    right_ear = obj.keypoint[sl.BODY_PARTS_POSE_18.RIGHT_EAR.value]
    nose = obj.keypoint[sl.BODY_PARTS_POSE_18.NOSE.value]
    gaze_vector = nose - (right_ear + left_ear) / 2
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
    obj.gaze_vector = gaze_vector

    # * angle between gaze vector and camera-nose vector
    nose_vector = nose / np.linalg.norm(nose)
    angle = np.arccos(np.clip(np.dot(nose_vector, gaze_vector), -1.0, 1.0))
    # radian to degree
    angle = angle * 180 / np.pi
    angle = 180 - angle
    obj.angle = angle


def get_palm_locations(obj):
    """returns a 128x128 image around the palms"""
    left_wrist = obj.keypoint_2d[sl.BODY_PARTS_POSE_18.LEFT_WRIST.value]
    # right_wrist = obj.keypoint_2d[sl.BODY_PARTS_POSE_18.RIGHT_WRIST.value]
    left_bb = [left_wrist[0], left_wrist[1]]
    return left_bb


def get_keypoints(obj):
    """returns a keypoint torch.tensor
    # A ------ B
    # | Object |
    # D ------ C
    """
    keypoints = []
    # format keypoint coordinates to relative position inside bounding box
    for kp in obj.keypoint_2d:
        keypoints.append(
            (kp[0] - obj.bounding_box_2d[0][0])
            / (obj.bounding_box_2d[1][0] - obj.bounding_box_2d[0][0]))
        keypoints.append(
            (kp[1] - obj.bounding_box_2d[0][1])
            / (obj.bounding_box_2d[3][1] - obj.bounding_box_2d[0][1]))

    keypoints_np = np.array(keypoints, dtype=np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keypoints_tensor = torch.tensor(keypoints_np)
    keypoints_tensor = keypoints_tensor.to(device)
    # print(obj.keypoint_2d)
    # print(keypoints)
    # print(obj.bounding_box_2d)
    return keypoints_tensor


if __name__ == "__main__":

    @timeit
    def test():
        x = list(range(6666))
        print(x)
    test()
