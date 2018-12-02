import numpy as np
import cv2
import math
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints

import argparse
import chainer
from entity import params

chainer.using_config('enable_backprop', False)

class FeatureExtractor:
    def __init__(self, pose_detector, hand_detector):
        self.pose_detector = pose_detector
        self.hand_detector = hand_detector

    def extractFeatures(self, img_path):
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

        # res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

        # each person detected
        for person_pose in person_pose_array:

            res["body"] = [list(k) for k in person_pose]

            unit_length = self.pose_detector.get_unit_length(person_pose)

            # hands estimation
            hands = self.pose_detector.crop_hands(img, person_pose, unit_length)
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = self.hand_detector(hand_img, hand_type="left")

                res["left"] = hand_keypoints
                res["left_bbox"] = bbox
                # res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = self.hand_detector(hand_img, hand_type="right")

                res["right"] = hand_keypoints
                res["right_bbox"] = bbox

                # res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                # cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        # print('Saving result into result.png...')
        # cv2.imwrite('result.png', res_img)

        return ExtractorOutput(res, img)

class ExtractorOutput:
    def __init__(self, keypoints, img):
        self.keypoints = keypoints
        self.img = img

    def get_representation(self):
        img = self.img
        person_pose_array = np.array([self.keypoints["body"]])
        print(person_pose_array)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

        hand_keypoints = self.keypoints["left"]
        bbox = self.keypoints["left_bbox"]
        res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
        cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        hand_keypoints = self.keypoints["right"]
        bbox = self.keypoints["right_bbox"]
        res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
        cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        return res_img

    def to_image(self, img_path):
        cv2.imwrite(img_path, self.get_representation())

    def get_keypoints(self):
        return Keypoints(self.keypoints["body"], self.keypoints["left"], self.keypoints["right"])

class Keypoints:
    def __init__(self, body, left, right):
        self.body = body
        self.left = left
        self.right = right

    def normalize(self):
        offset = self.body[1] # keypoint of the neck
        body = [[k[0] - offset[0], k[1] - offset[1], k[2]] for k in self.body]
        left = [[k[0] - offset[0], k[1] - offset[1], k[2]] for k in self.left]
        right = [[k[0] - offset[0], k[1] - offset[1], k[2]] for k in self.right]

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

print("Loading pose detection model...")
pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=0)
print("Loading hand detection model...")
hand_detector = HandDetector("handnet", "models/handnet.npz", device=0)

ext = FeatureExtractor(pose_detector, hand_detector)
