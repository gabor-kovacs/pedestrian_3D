import sys
import numpy as np
import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

import torch

# Action detection
from action_recognition.inference import init_model, detect

# Hand detection
from hand_detection.blazebase import resize_pad, denormalize_detections
from hand_detection.blazepalm import BlazePalm
from hand_detection.blazehand_landmark import BlazeHandLandmark
from hand_detection.visualization import draw_landmarks, HAND_CONNECTIONS

from lib.lib import get_gaze, get_keypoints, get_palm_locations


class People:
    """Class containing the list of tracked people"""
    def __init__(self, objects):
        self.timestamp = objects.timestamp
        self.object_list = [Person(obj) for obj in objects.object_list]
        self.is_new = objects.is_new
        self.is_tracked = objects.is_tracked
        self.get_object_data_from_id = objects.get_object_data_from_id


class Person:
    """Class containing info about one tracked person"""

    __slots__ = (
        "action_state",
        "bounding_box",
        "bounding_box_2d",
        "confidence",
        "dimensions",
        "global_root_orientation",
        "head_bounding_box",
        "head_bounding_box_2d",
        "head_position",
        "id",
        "keypoint",
        "keypoint_2d",
        "keypoint_confidence",
        "label",
        "local_orientation_per_joint",
        "local_position_per_joint",
        "mask",
        "position",
        "raw_label",
        "sublabel",
        "tracking_state",
        "unique_object_id",
        "velocity",
        "angle",
        "gaze_vector",
    )

    def __init__(self, object_data):
        # copy class attributes from pyzed object (no python dunder methods)
        for attr in dir(object_data):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(object_data, attr))
        # extend it with custom attributes
        self.angle = 0
        self.gaze_vector = np.array([0.0, 0.0, 0.0])


def main():
    """Real time mode"""
    print("Running object detection ... Press 'Esc' to quit")

    print(f"torch.cuda.is_available() {torch.cuda.is_available()}")

    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # HD1080 HD2K HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_maximum_distance = 20
    init_params.depth_minimum_distance = 1
    is_playback = False  # Defines if an SVO is used

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print(f"Using SVO file: {filepath}")
        init_params.svo_real_time_mode = False
        init_params.set_from_svo_file(filepath)
        is_playback = True

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))

        exit()

    # Enable positional tracking module
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static in space,
    # enabling this setting below provides better depth quality
    # and faster computation
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    # Smooth skeleton move
    obj_param.enable_body_fitting = True
    # Track people across images flow
    obj_param.enable_tracking = True
    # HUMAN_BODY_ACCURATE HUMAN_BODY_MEDIUM HUMAN_BODY_FAST
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE
    # Choose the BODY_FORMAT you wish to use
    obj_param.body_format = sl.BODY_FORMAT.POSE_18

    # Defines if the object detection will track objects across images flow.
    zed.enable_object_detection(obj_param)

    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(
        min(camera_infos.camera_resolution.width, 960),
        min(camera_infos.camera_resolution.height, 540),
    )
    point_cloud_highlight_res = sl.Resolution(
        min(camera_infos.camera_resolution.width, 96),
        min(camera_infos.camera_resolution.height, 54),
    )
    point_cloud_render = sl.Mat()
    point_cloud_highlight_render = sl.Mat()
    # viewer.init(
    #   camera_infos.camera_model,
    #   point_cloud_res,
    #   obj_param.enable_tracking
    # )
    viewer.init(
        sl.MODEL.ZED2,
        point_cloud_res,
        point_cloud_highlight_res,
        obj_param.enable_tracking,
        obj_param.body_format,
    )

    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Runtime parameters
    runtime_params = sl.RuntimeParameters()
    # runtime_params.confidence_threshold = 50

    # Create objects that will store SDK outputs
    point_cloud = sl.Mat(
        point_cloud_res.width,
        point_cloud_res.height,
        sl.MAT_TYPE.F32_C4,
        sl.MEM.CPU,
    )
    point_cloud_highlight = sl.Mat(
        point_cloud_highlight_res.width,
        point_cloud_highlight_res.height,
        sl.MAT_TYPE.F32_C4,
        sl.MEM.CPU,
    )

    objects = sl.Objects()
    image_left = sl.Mat()
    # image_full = sl.Mat()

    # Utilities for 2D display
    display_resolution = sl.Resolution(
        min(camera_infos.camera_resolution.width, 640),
        min(camera_infos.camera_resolution.height, 360),
    )
    image_scale = [
        display_resolution.width / camera_infos.camera_resolution.width,
        display_resolution.height / camera_infos.camera_resolution.height,
    ]
    image_left_ocv = np.full(
        (display_resolution.height, display_resolution.width, 4),
        [245, 239, 239, 255],
        np.uint8,
    )

    # Utilities for tracks view
    camera_config = zed.get_camera_information().camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(
        tracks_resolution,
        camera_config.camera_fps,
        init_params.depth_maximum_distance,
    )
    track_view_generator.set_camera_calibration(
        camera_config.calibration_parameters
    )
    image_track_ocv = np.zeros(
        (tracks_resolution.height, tracks_resolution.width, 4), np.uint8
    )

    # Will store the 2D image and tracklet views
    global_image = np.full(
        (
            display_resolution.height,
            display_resolution.width + tracks_resolution.width,
            4,
        ),
        [245, 239, 239, 255],
        np.uint8,
    )

    # Camera pose
    cam_w_pose = sl.Pose()
    # cam_c_pose = sl.Pose()

    quit_app = False

    model = init_model()

    # init hand detection stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    palm_detector = BlazePalm().to(device)
    palm_detector.load_weights("./hand_detection/blazepalm.pth")
    palm_detector.load_anchors("./hand_detection/anchors_palm.npy")
    palm_detector.min_score_thresh = 0.75
    hand_regressor = BlazeHandLandmark().to(device)
    hand_regressor.load_weights("./hand_detection/blazehand_landmark.pth")

    while viewer.is_available() and quit_app is False:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            # Retrieve image
            zed.retrieve_image(
                image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution
            )
            image_render_left = image_left.get_data()

            # print(f"{zed.get_current_fps():.2f} FPS")

            # Retrieve objects
            np.copyto(image_left_ocv, image_render_left)

            returned_state = zed.retrieve_objects(objects, obj_runtime_param)
            people = People(objects)

            hand_bboxes = []

            for person in people.object_list:
                get_gaze(person)
                joints = get_keypoints(person)
                action = detect(model, joints)
                print(action)

                left_bb = get_palm_locations(person)

                print(left_bb)
                hand_bboxes.append(left_bb)
                # image_left_ocv = cv2.rectangle(
                # image_left_ocv,
                # (left_bb[0], left_bb[2]),
                # (left_bbhackerman[1], left_bb[3]),
                # (255, 0, 0), 1)

            print(f"SHAPE: {image_left_ocv.shape}")
            # display_resolution.width
            img1, img2, scale, pad = resize_pad(image_left_ocv[:, :, :3])
            normalized_palm_detections = palm_detector.predict_on_image(img1)
            palm_detections = denormalize_detections(
                normalized_palm_detections, scale, pad
            )

            xc, yc, scale, theta = palm_detector.detection2roi(
                palm_detections.cpu()
            )
            img, affine2, box2 = hand_regressor.extract_roi(
                image_left_ocv[:, :, :3], xc, yc, theta, scale
            )
            flags2, handed2, normalized_landmarks2 = hand_regressor(
                img.to(device)
            )
            landmarks2 = hand_regressor.denormalize_landmarks(
                normalized_landmarks2.cpu(), affine2
            )
            # for i in range(len(flags2)):
            # for index, flag in enumerate(flags2):
            #     landmark = landmarks2[index]
            #     if flag > .5:
            #         draw_landmarks(
            #             image_left_ocv, landmark[:, : 2],
            #             HAND_CONNECTIONS, size=2)

            for flag, landmark in zip(flags2, landmarks2):
                if flag > 0.5:
                    draw_landmarks(
                        image_left_ocv,
                        landmark[:, :2],
                        HAND_CONNECTIONS,
                        size=2,
                    )

            # cv2.imshow("bruh", image_left_ocv)
            # cv2.waitKey(10)

            if returned_state == sl.ERROR_CODE.SUCCESS:
                # Retrieve point cloud
                zed.retrieve_measure(
                    point_cloud,
                    sl.MEASURE.XYZRGBA,
                    sl.MEM.CPU,
                    point_cloud_res,
                )
                zed.retrieve_measure(
                    point_cloud_highlight,
                    sl.MEASURE.XYZRGBA,
                    sl.MEM.CPU,
                    point_cloud_highlight_res,
                )

                point_cloud.copy_to(point_cloud_render)
                point_cloud_highlight.copy_to(point_cloud_highlight_render)

                # Get camera pose
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                UPDATE_RENDER_VIEW = True
                UPDATE_3D_VIEW = True
                UPDATE_TRACKING_VIEW = True

                pc = point_cloud_render.get_data()
                pc_hilite = point_cloud_highlight_render.get_data()

                # 3D rendering
                if UPDATE_3D_VIEW:
                    viewer.update_view(pc, pc_hilite, people)

                    # viewer.update_view(image, bodies)

                # 2D rendering
                if UPDATE_RENDER_VIEW:
                    # np.copyto(image_left_ocv,image_render_left)
                    cv_viewer.render_2D(
                        image_left_ocv,
                        image_scale,
                        people,
                        obj_param.enable_tracking,
                    )
                    global_image = cv2.hconcat(
                        [image_left_ocv, image_track_ocv]
                    )

                # Tracking view
                if UPDATE_TRACKING_VIEW:
                    track_view_generator.generate_view(
                        people, cam_w_pose, image_track_ocv, people.is_tracked
                    )

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            cv2.waitKey(10)

            """
            obj_param = sl.ObjectDetectionParameters()
            # Smooth skeleton move
            obj_param.enable_body_fitting = True
            # Track people across images flow
            obj_param.enable_tracking = True
            # HUMAN_BODY_ACCURATE HUMAN_BODY_MEDIUM HUMAN_BODY_FAST
            obj_param.detection_model = sl.DETECTION_MODEL.PERSON_HEAD_BOX
            # Choose the BODY_FORMAT you wish to use
            # obj_param.body_format = sl.BODY_FORMAT.POSE_34
            # Choose the BODY_FORMAT you wish to use
            obj_param.body_format = sl.BODY_FORMAT.POSE_34
            # if the object detection will track objects across images flow
            zed.enable_object_detection(obj_param)
            """

        if is_playback and (
            zed.get_svo_position() == zed.get_svo_number_of_frames() - 1
        ):
            print("End of SVO")
            quit_app = True

    cv2.destroyAllWindows()
    viewer.exit()
    image_left.free(sl.MEM.CPU)
    point_cloud.free(sl.MEM.CPU)
    point_cloud_render.free(sl.MEM.CPU)
    point_cloud_highlight.free(sl.MEM.CPU)
    point_cloud_highlight_render.free(sl.MEM.CPU)

    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()

    zed.close()


if __name__ == "__main__":
    main()
