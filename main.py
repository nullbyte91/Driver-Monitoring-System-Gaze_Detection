import logging as log
import sys
import numpy as np 

import os.path as osp
import cv2
import time

from argparse import ArgumentParser
from math import cos, sin, pi

from openvino.inference_engine import IENetwork

from utils.ie_module import InferenceContext
from core.face_detector import FaceDetector
from core.headPos_Estimator import HeadPosEstimator
from core.landmarks_detector import LandmarksDetector
DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-m_fd", "--mode_face_detection", required=True, type=str,
                        help="Path to an .xml file with a trained Face Detection model")               
    
    parser.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    parser.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")

    parser.add_argument("-m_hp", "--model_head_position", required=True, type=str,
                        help="Path to an .xml file with a trained Head Pose Estimation model") 
    parser.add_argument('-d_hp', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Head Position model (default: %(default)s)")
    parser.add_argument("-m_lm", "--model_landmark_regressor", required=True, type=str,
                        help="Path to an .xml file with a trained Head Pose Estimation model") 
    parser.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")
    
    parser.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    parser.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")
    parser.add_argument('-cw', '--crop_width', default=0, type=int,
                        help="(optional) Crop the input stream to this width " \
                        "(default: no crop). Both -cw and -ch parameters " \
                        "should be specified to use crop.")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    parser.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    parser.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    parser.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    parser.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    parser.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    return parser

class ProcessOnFrame:
    # Queue size will be used to put frames in a queue for
    # Inference Engine
    QUEUE_SIZE = 1
    

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_hp, args.d_lm])
        
        # Create a Inference Engine Context
        self.context = InferenceContext()
        context = self.context

        # Load OpenVino Plugin based on device selection
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        # Load face detection model on Inference Engine
        face_detector_net = self.load_model(args.mode_face_detection)
        
        # Load Headposition model on Inference Engine
        head_position_net = self.load_model(args.model_head_position)

        # Load Landmark regressor model on Inference Engine
        landmarks_net = self.load_model(args.model_landmark_regressor)

        # Configure Face detector [detection threshold, ROI Scale]
        self.face_detector = FaceDetector(face_detector_net,
                                    confidence_threshold=args.t_fd,
                                    roi_scale_factor=args.exp_r_fd)
        
        # Configure Head Pose Estimation
        self.head_estimator = HeadPosEstimator(head_position_net)

        # Configure Landmark regressor
        self.landmarks_detector = LandmarksDetector(landmarks_net)

        # Face detector 
        self.face_detector.deploy(args.d_fd, context)
        
        # Head Position Detector
        self.head_estimator.deploy(args.d_hp, context)

        # Landmark detector
        self.landmarks_detector.deploy(args.d_lm, context)
        
        log.info("Models are loaded")
    
    def load_model(self, model_path):
        """
        Initializing IENetwork(Inference Enginer) object from IR files:
        
        Args:
        Model path - This should contain both .xml and .bin file

        :return Instance of IENetwork class
        """
        
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        # Load model on IE
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        
        return model


    def frame_pre_process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)
        return frame

    def face_detector_process(self, frame):
        """
        Predict Face detection

        Args:
        The Input Frame

        :return roi [xmin, xmax, ymin, ymax]
        """
        frame = self.frame_pre_process(frame)

        # Clear Face detector from previous frame
        self.face_detector.clear()

        # When we use async IE use buffer by using Queue
        self.face_detector.start_async(frame)

        # Predict and return ROI
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        
        self.rois = rois
        return rois

    def head_position_estimator_process(self, frame):
        """
        Predict head_position

        Args:
        The Input Frame

        :return headPoseAngles[angle_y_fc, angle_p_fc, angle_2=r_fc]
        """
        frame = self.frame_pre_process(frame)

        # Clean Head Position detection from previous frame
        self.head_estimator.clear()

        # Predict and return head position[Yaw, Pitch, Roll]
        self.head_estimator.start_async(frame, self.rois)
        headPoseAngles = self.head_estimator.get_headposition()

        return headPoseAngles

    def face_landmark_detector_process(self, frame):
        frame = self.frame_pre_process(frame)

        # Clean Landmark detection from previous frame
        self.landmarks_detector.clear()

        # Predict and return landmark detection[left_eye, right_eye, nose_tip, 
        # left_lip_corner, right_lip_corner]
        self.landmarks_detector.start_async(frame, self.rois)
        landmarks = self.landmarks_detector.get_landmarks()

        return landmarks

    
class DriverMointoring:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}

    def __init__(self, args):
        self.frame_processor = ProcessOnFrame(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1
    
    def update_fps(self):
        """
        Calculate FPS
        """
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_detection_roi(self, frame, roi):
        """
        Draw Face detection bounding Box

        Args:
        frame: The Input Frame
        roi: [xmin, xmax, ymin, ymax]
        """
        for i in range(len(roi)):
            # Draw face ROI border
            cv2.rectangle(frame,
                        tuple(roi[i].position), tuple(roi[i].position + roi[i].size),
                        (0, 220, 0), 2)

    def draw_head_poistion_points(self, frame, points, rois):
        """
        Draw Head Position on frame
        Args:
        frame: The Input Frame
        points: [angle_y_fc, angle_p_fc, angle_2=r_fc]
        roi: [xmin, xmax, ymin, ymax]
        """
        # Here head_position_x --> angle_y_fc  # Yaw
        #      head_position_y --> angle_p_fc  # Pitch
        #      head_position_z --> angle_r_fc  # Roll
        cos_r = cos(points.head_position_z * pi / 180)
        sin_r = sin(points.head_position_z * pi / 180)
        sin_y = sin(points.head_position_x * pi / 180)
        cos_y = cos(points.head_position_x * pi / 180)
        sin_p = sin(points.head_position_y * pi / 180)
        cos_p = cos(points.head_position_y * pi / 180)
        
        xmin = rois[0].position[0]
        ymin = rois[0].position[1]
        xmax = rois[0].position[0] + rois[0].size[0]
        ymax = rois[0].position[1] + rois[0].size[1]
        x = int(( xmin+ xmax) / 2)
        y = int((ymin + ymax) / 2)

        angle_len = 80
        # center to right
        cv2.line(frame, (x,y), (x+int(angle_len*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(angle_len*cos_p*sin_r)), (0, 0, 255), thickness=3)
        
        # center to top
        cv2.line(frame, (x, y), (x+int(angle_len*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(angle_len*cos_p*cos_r)), (0, 255, 0), thickness=3)
        
        # center to forward
        cv2.line(frame, (x, y), (x + int(angle_len*sin_y*cos_p), y + int(angle_len*sin_p)), (255, 0, 0), thickness=3)
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    def createEyeBoundingBox(self, point_x, point_y, scale=1.8):
        size  = cv2.norm(np.float32(point_x) - point_y)
        width = scale * size
        height = width
        midpoint_x = (point_x[0] + point_y[0]) / 2
        midpoint_y = (point_x[1] + point_y[1]) / 2
        result_x  = midpoint_x - (width / 2)
        result_y  = midpoint_y - (height / 2)
        return [int(result_x), int(result_y), int(height), int(width)]

    def draw_detection_keypoints(self, frame, landmarks, roi):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]

        for point in keypoints:
            center = roi[0].position + roi[0].size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)
        
    def display_interactive_window(self, frame):
        """
        Display using CV Window
        
        Args:
        frame: The input frame
        """
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)

        cv2.imshow('Face recognition demo', frame)

    def should_stop_display(self):
        """
        Check exit key from user
        """
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        """
        Function to capture a frame from input stream, Pre-process,
        Predict, and Display

        Args:
        input_stream: The input file[Image, Video or Camera Node]
        output_stream: CV writer or CV window
        """

        self.input_stream = input_stream
        self.output_stream = output_stream

        # Loop input stream until frame is None
        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = DriverMointoring.center_crop(frame, self.input_crop)
            
            # Get Face detection
            detections = self.frame_processor.face_detector_process(frame)

            # Get head Position
            headPosition = self.frame_processor.head_position_estimator_process(frame)

            # Get face landmarks 
            landmarks = self.frame_processor.face_landmark_detector_process(frame)

            # Draw Face ROI detection
            self.draw_detection_roi(frame, detections)

            # Draw head position
            self.draw_head_poistion_points(frame, headPosition, detections)

            # Draw detection keypoints
            self.draw_detection_keypoints(frame, landmarks[0], detections)

            # Write on disk 
            if output_stream:
                output_stream.write(frame)
            
            # Display on CV Window
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break
            
            # Update FPS
            self.update_fps()
            self.frame_num += 1

    def run(self, args):
        """
        Driver function trigger all the function
        Args:
        args: Input args
        """
        # Open Input stream
        input_stream = DriverMointoring.open_input_stream(args.input)
        
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        
        # FPS init
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        
        # Get the Frame org size
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Get the frame count if its a video
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # Crop the image if the user define input W, H
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))

        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        
        # Writer or CV Window
        output_stream = DriverMointoring.open_output_stream(args.output, fps, frame_size)
        log.info("Input stream file opened")

        # Process on Input stream
        self.process(input_stream, output_stream)

        # Release Output stream if the writer selected
        if output_stream:
            output_stream.release()
        
        # Relese input stream[video or Camera node]
        if input_stream:
            input_stream.release()

        # Distroy CV Window
        cv2.destroyAllWindows()
    
    @staticmethod
    def center_crop(frame, crop_size):
        """
        Center the image in the view
        """
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]
    @staticmethod
    def open_input_stream(path):
        """
        Open the input stream
        """
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)
    
    @staticmethod
    def open_output_stream(path, fps, frame_size):
        """
        Open the output stream
        """
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream
        
def main():
    
    args = build_argparser().parse_args()
    
    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    driverMonitoring = DriverMointoring(args)
    driverMonitoring.run(args)

if __name__ == "__main__":
    main()