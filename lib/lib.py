import pyzed.sl as sl
import numpy as np

def get_gaze(obj):
    # create gaze defined from ears and nose
    left_ear = obj.keypoint[sl.BODY_PARTS_POSE_34.LEFT_EAR.value]
    right_ear = obj.keypoint[sl.BODY_PARTS_POSE_34.RIGHT_EAR.value]
    nose = obj.keypoint[sl.BODY_PARTS_POSE_34.NOSE.value]
    gaze_vector =  nose - (right_ear + left_ear) / 2 
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
    obj.gaze_vector = gaze_vector

    # * angle between gaze vector and camera-nose vector
    nose_vector = nose / np.linalg.norm(nose)
    angle = np.arccos(np.clip(np.dot(nose_vector, gaze_vector), -1.0, 1.0))
    # radian to degree
    angle = angle * 180 / np.pi
    angle = 180 - angle
    obj.angle = angle