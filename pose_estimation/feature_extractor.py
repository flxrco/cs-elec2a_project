import numpy as np
import cv2
import math
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints

import argparse
import chainer
from entity import params
import time

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

            def record_hand(hand_type):
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
            hands = self.pose_detector.crop_hands(img, person_pose, unit_length)
            if hands["left"] is not None:
                record_hand("left")

            if hands["right"] is not None:
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
            bbox = self.keypoints["%s_bbox" % hand_type]
            # hand_keypoints = [[k[0] - bbox[0], k[1] - bbox[1], k[2]] for k in self.keypoints[hand_type]]
            nonlocal res_img
            hand_keypoints = []
            for k in self.keypoints[hand_type]:
                if k is not None:
                    k = [k[0] - bbox[0], k[1] - bbox[1], k[2]]

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

    def normalize(self):
        offset = self.body[1] # keypoint of the neck

        def normalize_segment(keypoints_array):
            res = []
            for k in keypoints_array:
                if k is not None:
                    k = [k[0] - offset[0], k[1] - offset[1], k[2]]
                res.append(k)
            return res

        body = normalize_segment(self.body)
        left = normalize_segment(self.left)
        right = normalize_segment(self.right)

        return Keypoints(body, left, right)

    def to_dict(self):
        return {
            "body": self.body,
            "left": self.left,
            "right": self.right
        }

    def to_list(self):
        return [self.body, self.left, self.right]

    def distance_formula(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)



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

    keypoint_dict = output.get_keypoints().to_dict()
    print("RAW KEYPOINTS:")
    for key in keypoint_dict:
        print("%s KEYPOINTS" % key)
        for kp in keypoint_dict[key]:
            print(kp)
        print()

    if args.norm:
        norm_dict = output.get_keypoints().normalize().to_dict()
        print("RAW NORMALIZED:")
        for key in norm_dict:
            print("%s KEYPOINTS" % key)
            for kp in norm_dict[key]:
                print(kp)
            print()

    if args.dest is not None:
        print("Saving representation to %s" % args.dest)
        output.to_image(args.dest)
