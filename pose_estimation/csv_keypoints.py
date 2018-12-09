from feature_extractor import Keypoints
import csv
import numpy as np
from statistics import mode

LABELS = {
    "body": ["nose", "neck", "shoulder-r", "elbow-r", "wrist-r", "shoulder-l", "elbow-l", "wrist-l", "hip-r", "knee-r", "ankle-r", "hip-l", "knee-l", "ankle-l", "eye-r", "eye-l", "ear-r", "ear-l"],
    "hand": ["wirst", "thumb-palm", "thumb-proximal", "thumb-middle", "thumb-distal", "index-palm", "index-proximal", "index-middle", "index-distal", "middle-palm", "middle-proximal", "middle-middle", "middle-distal", "ring-palm", "ring-proximal", "ring-middle", "ring-distal", "pinky-palm", "pinky-proximal", "pinky-middle", "pinky-distal"]
}

DIMS = ["x", "y", "z"]

def csv_keypoints(csv_file):
    data_arr = []

    for row in csv.DictReader(open(csv_file, 'r')):
        coords_dict = {
            "body": {},
            "left": {},
            "right": {}
        }

        for key in row:
            split = key.split("_")

            if len(split) == 3:
                seg, kp, dim = tuple(split)
                
                if kp not in coords_dict[seg]:
                    coords_dict[seg][kp] = [0.0, 0.0, 0.0]

                coords_dict[seg][kp][DIMS.index(dim)] = float(row[key])

        def build_arr(label_arr, kp_dict):
            return [kp_dict[lb] for lb in label_arr]

        # body, left, right = tuple([build_arr(LABELS[key], coords_dict[key]) for key in ["body", "left", "right"]])
        body = build_arr(LABELS["body"], coords_dict["body"])
        left = build_arr(LABELS["hand"], coords_dict["left"])
        right = build_arr(LABELS["hand"], coords_dict["right"])

        data_arr.append((row["class"], row["actor"], Keypoints(body, left, right).normalize()))

    return data_arr


def normalize_dataset(tuple_arr):
    dim_range = range(len(DIMS))
    mapping = [("body", "body"), ("left", "hand"), ("right", "hand")]
    feat_coll = {}

    for (gesture, actor, keypoints) in tuple_arr:
        kp_list = keypoints.to_list()
        for (bd_seg, lb_class) in mapping:
            for i in range(len(LABELS[lb_class])):
                lb = LABELS[lb_class][i]
                if lb not in feat_coll:
                    feat_coll[lb] = {
                        "min": [99999.99, 99999.99, 99999.99],
                        "max": [-99999.99, -99999.99, -99999.99]
                    }

                feat_minmax = feat_coll[lb]
                dims_arr = kp_list[bd_seg][i]

                replace_check = {
                    "min": [dims_arr[j] < feat_minmax["min"][j] for j in dim_range],
                    "max": [dims_arr[j] > feat_minmax["max"][j] for j in dim_range]
                }

                for key in replace_check:
                    for i in dim_range:
                        if replace_check[key][i]:
                            feat_minmax[key][i] = dims_arr[i]

    res_arr = []

    for (gesture, actor, keypoints) in tuple_arr:
        kp_list = keypoints.to_list()
        kp_init_data = {
            "body": [],
            "left": [],
            "right": []
        }

        for (bd_seg, lb_class) in mapping:
            for i in range(len(LABELS[lb_class])):
                lb = LABELS[lb_class][i]
                feat_minmax = feat_coll[lb]

                init_dim_arr = [0.0, 0.0, 0.0]
                kp_dim_arr = kp_list[bd_seg][i]

                for j in dim_range:
                    den = (feat_minmax["max"][j] - feat_minmax["min"][j])
                    if den != 0.0:
                        init_dim_arr[j] = (kp_dim_arr[j] - feat_minmax["min"][j]) / den

                kp_init_data[bd_seg].append(init_dim_arr)

        res_arr.append((gesture, actor, Keypoints(kp_init_data["body"], kp_init_data["left"], kp_init_data["right"])))

    return res_arr

dataset = normalize_dataset(csv_keypoints("data.csv"))

for (gest, actor, keypoints) in dataset:
    print("Gesture: %s, Actor: %s" % (gest, actor))
    kp_list = keypoints.to_list()
    for seg in kp_list:
        print(seg, kp_list[seg])

def knn_test(test_index, neighbors):
    t_gest, t_act, t_kp = dataset[test_index]
    train = dataset[:test_index] + dataset[test_index + 1:]

    raw_list = []

    for i in range(len(train)):
        data = train[i]
        d_act, d_gest, d_kp = data
        dist = t_kp.compute_distance(d_kp)
        
        raw_list.append({'dist': dist, 'data': data})

    sorted_list = sorted(raw_list, key=lambda k: k['dist'])
    print("A: %f, B: %f" % (sorted_list[0]['dist'], sorted_list[len(sorted_list) - 1]['dist']))
    print("ACTUAL | Gesture: %s; Actor: %s" % (t_gest, t_act))
    classes = []
    for data_dict in sorted_list[:neighbors]:
        n_gest, n_act, n_kp = data_dict['data']
        classes.append(n_gest)
        print("NEIGHBOR | Gesture: %s; Actor: %s; Dist: %f" % (n_gest, n_act, data_dict['dist']))

    class_count = {}
    for c in classes:
        if c not in class_count:
            class_count[c] = 0
        class_count[c] += 1

    class_max = max(class_count, key=class_count.get)  # Just use 'min' instead of 'max' for minimum.
    print("Predicted: %s; Correct: %r" % (class_max, class_max == t_gest))

knn_test(67, 25)