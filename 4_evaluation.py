import numpy as np
from pathlib import Path

from hloc.utils.read_write_model import qvec2rotmat
from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    gt_path = Path("/media/seungjae/T7/NaverLabs/Dataset/naverlabs_dataset/HyundaiDepartmentStore/1F/release/validation/sensors/trajectories.txt")
    est_path = Path("outputs/naverlabs/HD_1F/localization_superpoint+superglue.txt")

    num_matches = {}

    est_pose = {}
    with open(est_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            image_name = line.split(" ")[0]            
            qvec = [float(x) for x in line.split(" ")[1:5]]
            tvec = [float(x) for x in line.split(" ")[5:8]]
            est_pose[image_name] = {"qvec": qvec, "tvec": tvec}
            num_matches[image_name] = float(line.split(" ")[9])/float(line.split(" ")[8])

    gt_pose = {}
    with open(gt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == "#":
                continue

            device_id = line.split(", ")[1].split("_")[0]
            timestamp = line.split(", ")[0]
            image_name = device_id + "_" + timestamp + ".jpg"
            qvec = [float(x) for x in line.split(",")[2:6]]
            tvec = [float(x) for x in line.split(",")[6:10]]
            gt_pose[image_name] = {"qvec": qvec, "tvec": tvec}

    first_area = 0
    second_area = 0
    third_area = 0

    rot_diffs = []
    tra_diffs = []

    x = []
    y = []

    relative_R = None
    relative_t = None
    for key in est_pose.keys():
        R_q = qvec2rotmat(est_pose[key]["qvec"])
        t_q = np.array(est_pose[key]["tvec"])

        R_gt = qvec2rotmat(gt_pose[key]["qvec"])
        t_gt = np.array(gt_pose[key]["tvec"])

        rot_diff = (np.trace(R_q @ R_gt.T) -1.0) / 2.0
        rot_diff = np.rad2deg(np.arccos( np.clip(rot_diff, -1.0, 1.0) ))
        tra_diff = np.linalg.norm(t_q - t_gt)

        print("key: ", key)
        print("R_q:\n", R_q, "\nR_gt:\n", R_gt)
        print("t_q: ", t_q, "\nt_gt: ", t_gt)
        print("rot_diff: ", rot_diff, "tra_diff: ", tra_diff)

        # if tra_diff < 500.0:
        x.append(tra_diff)
        y.append(num_matches[key])

        if rot_diff < 1 and tra_diff < 0.1:
            first_area += 1
        elif rot_diff < 2 and tra_diff < 0.25:
            second_area += 1
        elif rot_diff < 5 and tra_diff < 1.0:
            third_area += 1
        
        if rot_diff < 5 and tra_diff < 1.0:
            rot_diffs.append(rot_diff)
            tra_diffs.append(tra_diff)

    plt.scatter(x, y)
    plt.show()


    mean_rot_diff = np.mean(rot_diffs)
    mean_tra_diff = np.mean(tra_diffs)

    print("mean_rot_diff: ", mean_rot_diff)
    print("mean_tra_diff: ", mean_tra_diff)

    total = len(est_pose)
    print("1st: " , first_area , "/", total, (float)(100.0 * first_area  / total))
    print("2nd: ", second_area, "/", total, (float)(100.0 * second_area / total))
    print("3rd: " , third_area , "/", total, (float)(100.0 * third_area  / total))