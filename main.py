import sys
import time
import numpy as np
import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

from lib.lib import *

class People(): 
    def __init__(self, objects):
        self.timestamp = objects.timestamp
        self.object_list = [Person(obj) for obj in objects.object_list] 
        self.is_new  = objects.is_new 
        self.is_tracked  = objects.is_tracked 
        self.get_object_data_from_id  = objects.get_object_data_from_id 
        
class Person():
    def __init__(self, object_data):
        self.id = object_data.id
        self.unique_object_id = object_data.unique_object_id
        self.raw_label  = object_data.raw_label 
        self.label   = object_data.label  
        self.raw_label  = object_data.raw_label 
        self.sublabel  = object_data.sublabel 
        self.tracking_state  = object_data.tracking_state 
        self.action_state  = object_data.action_state 
        self.position  = object_data.position 
        self.velocity  = object_data.velocity 
        self.bounding_box  = object_data.bounding_box 
        self.bounding_box_2d  = object_data.bounding_box_2d 
        self.confidence  = object_data.confidence 
        self.mask  = object_data.mask 
        self.dimensions   = object_data.dimensions  
        self.keypoint = object_data.keypoint
        self.keypoint_2d  = object_data.keypoint_2d 
        self.head_bounding_box   = object_data.head_bounding_box  
        self.head_bounding_box_2d   = object_data.head_bounding_box_2d
        self.head_position   = object_data.head_position
        self.keypoint_confidence   = object_data.keypoint_confidence
        self.local_position_per_joint   = object_data.local_position_per_joint
        self.local_orientation_per_joint    = object_data.local_orientation_per_joint 
        self.global_root_orientation     = object_data.global_root_orientation  

        # * extend
        self.angle = 0
        self.gaze_vector = np.array([0.0, 0.0, 0.0])


if __name__ == "__main__":
    print("Running object detection ... Press 'Esc' to quit")
    zed = sl.Camera()
    
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080 # HD1080 HD2K HD720 
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_maximum_distance = 20
    init_params.depth_minimum_distance = 1
    is_playback = False                             # Defines if an SVO is used
        
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = False
        init_params.set_from_svo_file(filepath)
        is_playback = True

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()


    # Enable positional tracking module
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static in space, enabling this setting below provides better depth quality and faster computation
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)


    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE # HUMAN_BODY_ACCURATE HUMAN_BODY_MEDIUM HUMAN_BODY_FAST
    obj_param.body_format = sl.BODY_FORMAT.POSE_34 # Choose the BODY_FORMAT you wish to use
    # Defines if the object detection will track objects across images flow.
    zed.enable_object_detection(obj_param)

    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_infos.camera_resolution.width, 960), min(camera_infos.camera_resolution.height, 540)) 
    point_cloud_highlight_res = sl.Resolution(min(camera_infos.camera_resolution.width, 96), min(camera_infos.camera_resolution.height, 54)) 
    point_cloud_render = sl.Mat()
    point_cloud_highlight_render = sl.Mat()
    # viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    viewer.init(sl.MODEL.ZED2, point_cloud_res, point_cloud_highlight_res, obj_param.enable_tracking, obj_param.body_format)
    
    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Runtime parameters
    runtime_params = sl.RuntimeParameters()
    # runtime_params.confidence_threshold = 50

    # Create objects that will store SDK outputs
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud_highlight = sl.Mat(point_cloud_highlight_res.width, point_cloud_highlight_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    objects = sl.Objects()
    image_left = sl.Mat()

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_infos.camera_resolution.width, 640), min(camera_infos.camera_resolution.height, 360))
    image_scale = [display_resolution.width / camera_infos.camera_resolution.width
                 , display_resolution.height / camera_infos.camera_resolution.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239,255], np.uint8)

    # Utilities for tracks view
    camera_config = zed.get_camera_information().camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    # Will store the 2D image and tracklet views 
    global_image = np.full((display_resolution.height, display_resolution.width+tracks_resolution.width, 4), [245, 239, 239,255], np.uint8)

    # Camera pose
    cam_w_pose = sl.Pose()
    cam_c_pose = sl.Pose()

    quit_app = False

    while(viewer.is_available() and (quit_app == False)):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            # print(f"{zed.get_current_fps():.2f} FPS")

            # Retrieve objects
            returned_state = zed.retrieve_objects(objects, obj_runtime_param)
            people = People(objects)

            for person in people.object_list:
                get_gaze(person)
        
            if (returned_state == sl.ERROR_CODE.SUCCESS):
                # Retrieve point cloud
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, point_cloud_res)
                zed.retrieve_measure(point_cloud_highlight, sl.MEASURE.XYZRGBA,sl.MEM.CPU, point_cloud_highlight_res)


                point_cloud.copy_to(point_cloud_render)
                point_cloud_highlight.copy_to(point_cloud_highlight_render)
                # Retrieve image
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_render_left = image_left.get_data()
                # Get camera pose
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                update_render_view = True
                update_3d_view = True
                update_tracking_view = True

                pc = point_cloud_render.get_data()
                pc_hilite = point_cloud_highlight_render.get_data()



                # 3D rendering
                if update_3d_view:
                    viewer.update_view(pc, pc_hilite, people)

                    # viewer.update_view(image, bodies) 

                # 2D rendering
                if update_render_view:
                    np.copyto(image_left_ocv,image_render_left)
                    cv_viewer.render_2D(image_left_ocv,image_scale, people, obj_param.enable_tracking)
                    global_image = cv2.hconcat([image_left_ocv,image_track_ocv])

                # Tracking view
                if update_tracking_view:
                    track_view_generator.generate_view(people, cam_w_pose, image_track_ocv, people.is_tracked)
                    
            cv2.imshow("ZED | 2D View and Birds View", global_image)
            cv2.waitKey(10)


            obj_param = sl.ObjectDetectionParameters()
            obj_param.enable_body_fitting = True            # Smooth skeleton move
            obj_param.enable_tracking = True                # Track people across images flow
            obj_param.detection_model = sl.DETECTION_MODEL.PERSON_HEAD_BOX # HUMAN_BODY_ACCURATE HUMAN_BODY_MEDIUM HUMAN_BODY_FAST
            obj_param.body_format = sl.BODY_FORMAT.POSE_34 # Choose the BODY_FORMAT you wish to use
            # Defines if the object detection will track objects across images flow.
            zed.enable_object_detection(obj_param)

            



        if (is_playback and (zed.get_svo_position() == zed.get_svo_number_of_frames()-1)):
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