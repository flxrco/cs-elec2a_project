import feature_extractor

LABELS = {
    "body": ["nose", "neck", "shoulder-r", "elbow-r", "wrist-r", "shoulder-l", "elbow-l", "wrist-l", "hip-r", "knee-r", "ankle-r", "hip-l", "knee-l", "ankle-l", "eye-r", "eye-l", "ear-r", "ear-l"],
    "hand": ["wirst", "thumb-palm", "thumb-proximal", "thumb-middle", "thumb-distal", "index-palm", "index-proximal", "index-middle", "index-distal", "middle-palm", "middle-proximal", "middle-middle", "middle-distal", "ring-palm", "ring-proximal", "ring-middle", "ring-distal", "pinky-palm", "pinky-proximal", "pinky-middle", "pinky-distal"]
}

DIMS = ["x", "y", "z"]

def normalize_dataset(tuple_arr):
    dims_range = range(len(DIMS))
    feat_coll = {}

    for (gesture, actor, keypoints) in tuple_arr:
        kp_list = keypoints.to_list()
        for (bd_seg, lb_class) in [("body", "body"), ("left", "hand"), ("right", "hand")]:
            for i in range(len(LABELS[lb_class])):
                lb = LABELS[lb_class]
                if lb not in feat_coll:
                    feat_coll[lb] = {
                        "min": [0.0, 0.0, 0.0],
                        "max": [0.0, 0.0, 0.0]
                    }

                feat_minmax = feat_coll[lb]
                dims_arr = kp_list[bd_seg][i]

                replace_check = {
                    "min": [dims_arr[j] < feat_minmax["min"] for j in dims_range],
                    "max": [dims_arr[j] > feat_minmax["max"] for j in dims_range]
                }

                for key in replace_check:
                    for i in dims_range:
                        if replace_check[key][i]:
                            feat_minmax[key][i] = dims_arr[i]

    print(feat_coll)