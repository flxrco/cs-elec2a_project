from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
from feature_extractor import FeatureExtractor, ExtractorOutput, Keypoints, build_template

import argparse
from entity import params
import pandas as pd
import time
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument('--dir', help='Image source dir')
    parser.add_argument('--imgdest', default=None, help='Representation img dest dir')
    parser.add_argument('--csvdest', help='.csv file dest filepath')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    start_time = time.time()
    print("Loading models...")
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    print("Loading complete. Time elapsed: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    extractor = FeatureExtractor(pose_detector, hand_detector)
    data = Keypoints.build_template()
    data["class"] = []
    data["actor"] = []

    img_no = len(os.listdir(args.dir))
    img_ctr = 0
    print("%d images to extract. Beginning extraction." % img_no)

    for img in os.listdir(args.dir):
        img_ctr += 1
        start_time = time.time()
        out = extractor.extract_features(os.path.join(args.dir, img))
        print("%d/%d extracted. Time elapsed: %s" % (img_ctr, img_no, time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

        if args.imgdest is not None:
            out.to_image(os.path.join(args.imgdest, img))
        
        kp = out.get_keypoints().to_flat_dict()
        details = img.split(".")[0].split("_")
        kp["class"] = [details[0]]
        kp["actor"] = [details[1]]

        for key in data:
            data[key].append(kp[key][0])

    pd.DataFrame.from_dict(data).to_csv(args.csvdest)