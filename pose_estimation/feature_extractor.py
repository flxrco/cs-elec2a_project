import numpy as np
import pandas as pd
import cv2
import math
import pprint
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints

import argparse
import chainer
from entity import params
import time

LABELS = {
    "body": ["nose", "neck", "shoulder-r", "elbow-r", "wrist-r", "shoulder-l", "elbow-l", "wrist-l", "hip-r", "knee-r", "ankle-r", "hip-l", "knee-l", "ankle-l", "eye-r", "eye-l", "ear-r", "ear-l"],
    "hand": ["wirst", "thumb-palm", "thumb-proximal", "thumb-middle", "thumb-distal", "index-palm", "index-proximal", "index-middle", "index-distal", "middle-palm", "middle-proximal", "middle-middle", "middle-distal", "ring-palm", "ring-proximal", "ring-middle", "ring-distal", "pinky-palm", "pinky-proximal", "pinky-middle", "pinky-distal"]
}

DIMS = ["x", "y", "z"]

chainer.using_config('enable_backprop', False)

class FeatureExtractor:

    def __init__(self, pose_detector, hand_detector):
        self.pose_detector = pose_detector
        self.hand_detector = hand_detector

    def extract_features(self, img_path):
        res = {
            "body": None,
            "left": None,
            "left_bbox": None,
            "right": None,
            "right_bbox": None
        }

        img = cv2.imread(img_path)

        person_pose_array, _ = self.pose_detector(img)

        if len(person_pose_array) > 1:
            person_pose_array = np.array([person_pose_array[0]])

        # each person detected
        for person_pose in person_pose_array:

            res["body"] = [list(k) for k in person_pose]
            unit_length = self.pose_detector.get_unit_length(person_pose)
            hands = self.pose_detector.crop_hands(img, person_pose, unit_length)

            def record_hand(hand_type):
                if hands[hand_type] is not None:
                    hand_img = hands[hand_type]["img"]
                    bbox = hands[hand_type]["bbox"]
                    hand_keypoints = self.hand_detector(hand_img, hand_type=hand_type)
                    res[hand_type] = []

                    for k in hand_keypoints:
                        if k is not None:
                            k = [k[0] + bbox[0], k[1] + bbox[1], k[2]]

                        res[hand_type].append(k)

                    res["%s_bbox" % hand_type] = bbox

            # hands estimation
            record_hand("left")
            record_hand("right")

        return ExtractorOutput(res, img)

class ExtractorOutput:

    def __init__(self, keypoints, img):
        self.keypoints = keypoints
        self.img = img

    def render_visualization(self):
        img = self.img
        person_pose_array = np.array([self.keypoints["body"]])
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

        def represent_hand(hand_type):
            
            if self.keypoints[hand_type] is not None:
                bbox = self.keypoints["%s_bbox" % hand_type]
                # hand_keypoints = [[k[0] - bbox[0], k[1] - bbox[1], k[2]] for k in self.keypoints[hand_type]]
                nonlocal res_img
                hand_keypoints = []
                for k in self.keypoints[hand_type]:
                    if k is not None:
                        if sum(k) != 0:
                            k = [k[0] - bbox[0], k[1] - bbox[1], k[2]]
                        else:
                            k = None

                    hand_keypoints.append(k)
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        represent_hand("left")
        represent_hand("right")

        return res_img

    def to_image(self, img_path):
        cv2.imwrite(img_path, self.render_visualization())

    def get_keypoints(self):
        return Keypoints(self.keypoints["body"], self.keypoints["left"], self.keypoints["right"])

class Keypoints:

    def __init__(self, body, left, right):
        self.body = body
        self.left = left
        self.right = right

        if self.left is None:
            self.left = [None for i in range(len(LABELS["hand"]))]

        if self.right is None:
            self.right = [None for i in range(len(LABELS["hand"]))]

        for i in range(len(LABELS["hand"])):
            if self.right[i] is None:
                self.right[i] = [0.0, 0.0, 0.0]

            if self.left[i] is None:
                self.left[i] = [0.0, 0.0, 0.0]


    def normalize(self):
        def normalize_segment(keypoints_array, offset):
            res = []
            for k in keypoints_array:
                if k is not None:
                    if sum(k) != 0.0:
                        k = [k[0] - offset[0], k[1] - offset[1], k[2]]
                res.append(k)
            return res

        body = normalize_segment(self.body, self.body[1])
        left = normalize_segment(self.left, self.left[0])
        right = normalize_segment(self.right, self.right[0])

        return Keypoints(body, left, right)

    def rescale(self, body_width=500, left_width=500, right_width=500):
        kp_list = self.to_list()
        
        new_kp = {
            "body": [],
            "left": [],
            "right": []
        }

        widths = {
            "body": body_width,
            "left": left_width,
            "right": right_width
        }

        coords_bbox = {}

        for key in kp_list:
            coords_bbox[key] = {}

            coords_bbox[key]["min"] = (
                min([kp[0] for kp in kp_list[key]]),
                min([kp[1] for kp in kp_list[key]])
            )

            coords_bbox[key]["max"] = (
                max([kp[0] for kp in kp_list[key]]),
                max([kp[1] for kp in kp_list[key]])
            )

        def calculate_bbox(min_dims, max_dims):
            min_x, min_y = min_dims
            max_x, max_y = max_dims

            return (abs(max_x - min_x), abs(max_y - min_y)) # (width, height)

        for key in kp_list:
            new_width = widths[key]
            coords = coords_bbox[key]
            width, height = calculate_bbox(coords["min"], coords["max"])
            
            if width == 0 and height == 0:
                for i in range(len(kp_list[key])):
                    new_kp[key].append([0.0, 0.0, 0.0])
                continue

            new_height = height * (width / new_width)

            scale_factor = [
                float(new_height) / height,
                float(new_width) / width
            ]

            for kp in kp_list[key]:
                new = [0, 0, kp[2]]
                for i in range(len(scale_factor)):
                    new[i] = kp[i] * scale_factor[i]
                new_kp[key].append(new)

        return Keypoints(new_kp["body"], new_kp["left"], new_kp["right"])


    def to_dict(self):
        res = {
            "body": {},
            "left": {},
            "right": {}
        }

        def build_dict(kp_arr, label_arr, res_dict):
            for i in range(len(kp_arr)):
                res_dict[label_arr[i]] = kp_arr[i]

        build_dict(self.body, LABELS["body"], res["body"])
        build_dict(self.right, LABELS["hand"], res["right"])
        build_dict(self.left, LABELS["hand"], res["left"])

        return res

    def to_list(self):
        return {
            "body": self.body,
            "left": self.left,
            "right": self.right
        }

    def to_flat_dict(self):
        flat_dict = {}
        kp_dict = self.to_dict()

        for key in kp_dict:
            for label in kp_dict[key]:
                coords = kp_dict[key][label]
                for i in range(len(DIMS)):
                    flat_dict["%s_%s_%s" % (key, label, DIMS[i])] = [round(float(coords[i]), 2)]


        return flat_dict

    def compute_distance(self, kp):
        body = [Keypoints.distance_formula(self.body[i], kp.body[i]) for i in range(len(LABELS["body"]))]
        left = [Keypoints.distance_formula(self.left[i], kp.left[i]) for i in range(len(LABELS["hand"]))]
        right = [Keypoints.distance_formula(self.right[i], kp.right[i]) for i in range(len(LABELS["hand"]))]
        return 3 * (sum(left) + sum(right)) + sum(body)

    def distance_formula(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def build_template():
        template = {}

        def build_dict(label_arr, prefix):
            for label in label_arr:
                for dim in ["x", "y", "z"]:
                    template["%s_%s_%s" % (prefix, label, dim)] = []

        build_dict(LABELS["body"], "body")
        build_dict(LABELS["hand"], "left")
        build_dict(LABELS["hand"], "right")

        return template

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument('--src', help='Source image filepath')
    parser.add_argument('--dest', default=None, help='Destimation image filepath')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize keypoints')
    args = parser.parse_args()

    start_time = time.time()
    print("Loading models...")
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    print("Loading complete. Time elapsed: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    extractor = FeatureExtractor(pose_detector, hand_detector)

    start_time = time.time()
    print("Extracting...")
    output = extractor.extract_features(args.src)
    print("Extraction complete. Time elapsed: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    keypoints = output.get_keypoints()

    if args.norm:
        keypoints = keypoints.normalize()

    print(keypoints.to_flat_dict())

    if args.dest is not None:
        print("Saving representation to %s" % args.dest)
        output.to_image(args.dest)

    print("Elapsed: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
