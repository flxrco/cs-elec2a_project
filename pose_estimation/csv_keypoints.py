from feature_extractor import Keypoints
import csv
import numpy as np
from statistics import mode
import random

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

        kp = Keypoints(body, left, right)

        data_arr.append((row["class"], row["actor"], Keypoints(body, left, right).normalize().rescale()))

    return data_arr

def normalize_dataset(tuple_arr):
    dim_range = range(len(DIMS))
    mapping = [("body", "body"), ("left", "hand"), ("right", "hand")]
    
    minmax_coll = {
        "body": {},
        "left": {},
        "right": {}
    }

    for (gesture, actor, keypoints) in tuple_arr:
        kp_list = keypoints.to_list()
        for (bd_seg, lb_class) in mapping:
            for i in range(len(LABELS[lb_class])):
                lb = LABELS[lb_class][i]
                
                if lb not in minmax_coll[bd_seg]:
                    minmax_coll[bd_seg][lb] = {
                        "min": [99999.99, 99999.99, 99999.99],
                        "max": [-99999.99, -99999.99, -99999.99]
                    }

                minmax = minmax_coll[bd_seg][lb] # {'min': [x, y, z], 'max': [x, y, z]}
                kp = kp_list[bd_seg][i] # [x, y, z]

                rep = {
                    "min": [kp[i] < minmax["min"][i] for i in dim_range],
                    "max": [kp[i] > minmax["max"][i] for i in dim_range]
                }

                for key in rep:
                    for i in dim_range:
                        if rep[key][i]:
                            minmax[key][i] = kp[i]

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
                minmax = minmax_coll[bd_seg][lb]

                init_dim_arr = [0.0, 0.0, 0.0]
                kp_dim_arr = kp_list[bd_seg][i]

                for j in dim_range:
                    den = (minmax["max"][j] - minmax["min"][j])
                    if den != 0.0:
                        init_dim_arr[j] = (kp_dim_arr[j] - minmax["min"][j]) / den

                kp_init_data[bd_seg].append(init_dim_arr)

        res_arr.append((gesture, actor, Keypoints(kp_init_data["body"], kp_init_data["left"], kp_init_data["right"])))

    return res_arr

# def normalize_dataset(tuple_arr):
#     dim_range = range(len(DIMS))
#     mapping = [("body", "body"), ("left", "hand"), ("right", "hand")]
#     feat_coll = {}

#     for (gesture, actor, keypoints) in tuple_arr:
#         kp_list = keypoints.to_list()
#         for (bd_seg, lb_class) in mapping:
#             for i in range(len(LABELS[lb_class])):
#                 lb = LABELS[lb_class][i]
#                 if lb not in feat_coll:
#                     feat_coll[lb] = {
#                         "min": [99999.99, 99999.99, 99999.99],
#                         "max": [-99999.99, -99999.99, -99999.99]
#                     }

#                 feat_minmax = feat_coll[lb]
#                 dims_arr = kp_list[bd_seg][i]

#                 replace_check = {
#                     "min": [dims_arr[j] < feat_minmax["min"][j] for j in dim_range],
#                     "max": [dims_arr[j] > feat_minmax["max"][j] for j in dim_range]
#                 }

#                 for key in replace_check:
#                     for i in dim_range:
#                         if replace_check[key][i]:
#                             feat_minmax[key][i] = dims_arr[i]

#     res_arr = []

#     for (gesture, actor, keypoints) in tuple_arr:
#         kp_list = keypoints.to_list()
#         kp_init_data = {
#             "body": [],
#             "left": [],
#             "right": []
#         }

#         for (bd_seg, lb_class) in mapping:
#             for i in range(len(LABELS[lb_class])):
#                 lb = LABELS[lb_class][i]
#                 feat_minmax = feat_coll[lb]

#                 init_dim_arr = [0.0, 0.0, 0.0]
#                 kp_dim_arr = kp_list[bd_seg][i]

#                 for j in dim_range:
#                     den = (feat_minmax["max"][j] - feat_minmax["min"][j])
#                     if den != 0.0:
#                         init_dim_arr[j] = (kp_dim_arr[j] - feat_minmax["min"][j]) / den

#                 kp_init_data[bd_seg].append(init_dim_arr)

#         res_arr.append((gesture, actor, Keypoints(kp_init_data["body"], kp_init_data["left"], kp_init_data["right"])))

#     return res_arr

dataset = normalize_dataset(csv_keypoints("data.csv"))

# for (gest, actor, keypoints) in dataset:
#     print("Gesture: %s, Actor: %s" % (gest, actor))
#     kp_list = keypoints.to_list()
#     for seg in kp_list:
#         print(seg, kp_list[seg])

# def knn_test(test_index, neighbors):
#     t_gest, t_act, t_kp = dataset[test_index]
#     train = dataset[:test_index] + dataset[test_index + 1:]

#     raw_list = []

#     for i in range(len(train)):
#         data = train[i]
#         d_act, d_gest, d_kp = data
#         dist = t_kp.compute_distance(d_kp)
        
#         raw_list.append({'dist': dist, 'data': data})

#     sorted_list = sorted(raw_list, key=lambda k: k['dist'])
#     # print("A: %f, B: %f" % (sorted_list[0]['dist'], sorted_list[len(sorted_list) - 1]['dist']))
#     # print("ACTUAL | Gesture: %s; Actor: %s" % (t_gest, t_act))
#     classes = []
#     for data_dict in sorted_list[:neighbors]:
#         n_gest, n_act, n_kp = data_dict['data']
#         classes.append(n_gest)
#         # print("NEIGHBOR | Gesture: %s; Actor: %s; Dist: %f" % (n_gest, n_act, data_dict['dist']))

#     class_count = {}
#     for c in classes:
#         if c not in class_count:
#             class_count[c] = 0
#         class_count[c] += 1

#     class_max = max(class_count, key=class_count.get)  # Just use 'min' instead of 'max' for minimum.
#     # print("Predicted: %s; Correct: %r" % (class_max, class_max == t_gest))
#     # print("Actual: %s; Predicted: %s; Correct?: %r" % (t_gest, class_max, class_max == t_gest))
#     return class_max == t_gest

# accu_dict = {}
# for k in range(1, 26):
#     # print("%d neighbors." % k)
#     ctr = 0
#     for i in range(len(dataset)):
#         if knn_test(i, k):
#             ctr += 1
#     # print("Accuracy: %f" % (100 * (ctr/len(dataset))))
#     print("Neighbors: %d, Accuracy: %f" % (k, 100 * (ctr/len(dataset))))

def train_test_split(dataset, train_ratio):
    ds_coll = {}
    for (gest, act, kp) in dataset:
        if gest not in ds_coll:
            ds_coll[gest] = []
        ds_coll[gest].append((gest, act, kp))

    train = []
    test = []

    for gest in ds_coll:
        gest_clone = [g for g in ds_coll[gest]]
        train_count = int((float(train_ratio)/100) * len(ds_coll[gest]))
        for i in range(train_count):
            victim = random.randint(0, len(gest_clone) - 1)
            train.append(gest_clone[victim])
            del gest_clone[victim]
        for g in gest_clone:
            test.append(g)

    return (train, test)

train, test = train_test_split(dataset, 80)

def knn_test(train_set, test, neighbor_count = 5):

    a_gest, a_act, a_kp = test

    raw_list = []
    for (t_gest, t_act, t_kp) in train_set:
        raw_list.append({"dist": t_kp.compute_distance(a_kp), "data": (t_gest, t_act, t_kp)})

    sorted_list = sorted(raw_list, key=lambda k: k["dist"])
    classes = {}
    # print("ACTUAL | Gesture: %s; Actor: %s" % (a_gest, a_act))
    for data_dict in sorted_list[:neighbor_count]:
        n_gest, n_act, n_kp = data_dict['data']
        if n_gest not in classes:
            classes[n_gest] = 0
        classes[n_gest] += 1
        # print("NEIGHBOR | Gesture: %s; Actor: %s; Dist: %f" % (n_gest, n_act, data_dict['dist']))

    class_max = max(classes, key=classes.get)

    return class_max == a_gest

ctr = 0
accu = {}
for t in test:
    t_gest, t_act, t_kp = t
    if t_gest not in accu:
        accu[t_gest] = [0, 0] # pos 0 corr, pos 1 total
    accu[t_gest][1] += 1
    if knn_test(train, t, 6):
        ctr += 1
        accu[t_gest][0] += 1
print("Total accuracy: %0.2f" % (float(ctr)/len(test) * 100))
for a in accu:
    print("%s: %0.2f" % (a, (float(accu[a][0])/accu[a][1]) * 100))
        