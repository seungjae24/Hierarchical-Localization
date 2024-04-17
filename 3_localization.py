import torch

from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_gt_poses,
    logger,
)

from hloc import extractors, matchers
from hloc.utils.base_model import dynamic_load
from hloc.utils.read_write_model import qvec2rotmat
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster, do_covisibility_clustering

import numpy as np
import pycolmap
from tqdm import tqdm
import pickle
import cv2
import os

from common_util import configs

def read_intrinsics(intrinsic_path):
    intrinsics = {}
    with open(intrinsic_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            else:
                device_id = line.split(",")[0].split("_")[0]
                model = line.split(",")[3][1:]
                width = line.split(",")[4][1:]
                height = line.split(",")[5][1:]
                params = [x[1:] for x in line.split(",")[6:]]
                params = [float(x) for x in params]

                print(device_id, model, width, height, params)

                intrinsics[device_id] = {"model": model, "width": width, "height": height, "params": params}
    
    return intrinsics

def get_poses(pose_path, format=".jpg"):
    poses = {}
    with open(pose_path, "r") as pf:
        lines = pf.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            elif "lidar" in line.split(",")[1]:
                continue
            else:
                timestamp = line.split(", ")[0]
                device_id = line.split(", ")[1].split("_")[0]
                id = device_id + "_" + timestamp + format
                qvec = [float(x) for x in line.split(",")[2:6]]
                tvec = [float(x) for x in line.split(",")[6:9]]
                poses[id] = {"qvec": qvec, "tvec": tvec}
    
    return poses

def find_closest_db_images_for_q(q_poses, db_poses, match_pair_path, num_closest=30):
    
    q_db_pairs = {}
    
    rewrite_pairs = False
    if match_pair_path.exists() and rewrite_pairs == False:
        with open(match_pair_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                q_id = line.split(" ")[0]
                db_ids = line.split(" ")[1].split("\n")[0]
                if q_id in q_db_pairs:
                    q_db_pairs[q_id].append(db_ids)
                else:
                    q_db_pairs[q_id] = [db_ids]
        return q_db_pairs
    else:
        with open(match_pair_path, "w") as f:
            for q_id in tqdm(q_poses.keys()):
                q_pose = q_poses[q_id]

                R_q = qvec2rotmat(q_pose["qvec"])
                t_q = np.array(q_pose["tvec"])

                tra_diffs = []

                for i, db_id in enumerate(db_poses.keys()):
                    db_pose = db_poses[db_id]
                    R_db = qvec2rotmat(db_pose["qvec"])
                    t_db = np.array(db_pose["tvec"])

                    # compute distance between two poses
                    rot_diff = np.rad2deg(np.arccos(np.clip(np.trace(R_q @ R_db.T) - 1, -1, 1)))
                    if rot_diff > 30:
                        continue
                    
                    tra_diff = np.linalg.norm(t_q - t_db)
                    tra_diffs.append((db_id, tra_diff))

                tra_diffs.sort(key=lambda x: x[1])
                min_num = min(num_closest, len(tra_diffs))

                q_db_pairs[q_id] = [ x[0] for x in tra_diffs[:min_num] ]
                
                for db_id in q_db_pairs[q_id]:
                    f.write(q_id + " " + db_id + "\n")

    return q_db_pairs

def debugging(q_db_pairs, q_image_path, db_image_path):

    mean_pair = 0
    for key in q_db_pairs.keys():
        print("Query image:", key)

        q_img_file = q_image_path + key
        q_img = cv2.imread(q_img_file)

        for i in range(len(q_db_pairs[key])):
            print("DB image:", q_db_pairs[key][i])

            db_img_file = db_image_path + q_db_pairs[key][i]
            db_img = cv2.imread(db_img_file)

            q_img = cv2.resize(q_img, (640, 480))
            db_img = cv2.resize(db_img, (640, 480))

            img = np.concatenate((q_img, db_img), axis=1)
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", img)
            cv2.waitKey(200)

            if i == 5:
                break

        mean_pair += len(q_db_pairs[key])
    mean_pair /= len(q_db_pairs)
    logger.info("Mean number of pairs:%d", mean_pair)

# def getImageFiles(txt_file):

#     image_files = {}
#     with open(txt_file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             if line.startswith("#"):
#                 continue
#             else:
#                 timestamp = line.split(", ")[0]
#                 device_id = line.split(", ")[1].split("_")[0]
#                 id = device_id + "_" + timestamp + ".jpg"
#                 image_files[id] = line.split(", ")[2]

#     return image_files

def getColmapCameras(q_db_pairs, q_intrinsics):

    queries = []
    for q_id in q_db_pairs.keys():
        q_cam = getColmapCamera(q_id, q_intrinsics)
        queries.append((q_id, q_cam))
    return queries

def getColmapCamera(qname, q_intrinsics):

    device_id = qname.split("/")[-1].split("_")[0]
    q_intr = q_intrinsics[device_id]

    model = q_intr["model"]
    width = q_intr["width"]
    height = q_intr["height"]
    params = q_intr["params"]

    q_cam = pycolmap.Camera(model=model, width=int(width), height=int(height), params=params)

    return q_cam

def convertToDictionary(match_pairs):
    q_db_pairs = {}
    with open(match_pairs, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            q_id = line.split(" ")[0]
            db_id = line.split(" ")[1]
            if q_id in q_db_pairs:
                q_db_pairs[q_id].append(db_id)
            else:
                q_db_pairs[q_id] = [db_id]
    return q_db_pairs

def write_results(cam_from_world, num_matches, results, prepend_camera_name):
    with open(results, "w") as f:
        for query, t in cam_from_world.items():
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec} {num_matches[query][0]} {num_matches[query][1]}\n")

def write_logs(logs, results):
    
    logs_path = f"{results}_logs.pkl"
    print(f"Writing logs to {logs_path}...")

    # TODO: Resolve pickling issue with pycolmap objects.
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)

def main():

    seq                     = "HD_1F"
    data_path               = configs[seq]["data_path"]
    
    q_image_path            = data_path + "/validation/sensors/records_data/"
    q_pose_path             = data_path + "/validation/sensors/trajectories.txt"
    q_image_intrinsic_path  = data_path + "/validation/sensors/sensors.txt"

    db_image_path           = data_path + "/mapping_with_lidar/sensors/records_data/"
    db_pose_path            = data_path + "/mapping_with_lidar/sensors/trajectories.txt"

    # q_image_files           = getImageFiles(data_path + "/validation/sensors/records_camera.txt")
    # db_image_files          = getImageFiles(data_path + "/mapping_with_lidar/sensors/records_camera.txt")
    
    outputs     = Path("outputs/naverlabs/" + seq)
    sfm_dir     = outputs / "sfm_superpoint+superglue"
    match_pairs = outputs / "pairs-q_db.txt"
    results     = outputs / "localization_superpoint+superglue.txt"

    q_intrinsics = read_intrinsics(q_image_intrinsic_path)

    db_retrieval_conf  = extract_features.confs["netvlad-db"]
    q_retrieval_conf   = extract_features.confs["netvlad-q"]
    feature_conf    = extract_features.confs["superpoint_nv"]
    matcher_conf    = match_features.confs["superglue-loc"]

    # for each query image, find the closest 20 images in db image
    # q_poses     = get_poses(q_pose_path, format=".jpg")
    # db_poses    = get_poses(db_pose_path, format=".jpg")
    # q_db_pairs  = find_closest_db_images_for_q(q_poses, db_poses, match_pairs)
    # debugging(q_db_pairs, q_image_path, db_image_path, q_image_files, db_image_files)

    db_retrieval    = extract_features.main(db_retrieval_conf, Path(db_image_path), outputs)
    q_retrieval     = extract_features.main(q_retrieval_conf, Path(q_image_path), outputs)
    pairs_from_retrieval.main(q_retrieval, match_pairs, num_matched=30, db_descriptors=db_retrieval)

    q_db_pairs = convertToDictionary(match_pairs)
    # debugging(q_db_pairs, q_image_path, db_image_path)

    features    = extract_features.main(feature_conf, Path(q_image_path), outputs)
    features    = extract_features.main(feature_conf, Path(db_image_path), outputs)
    matches     = match_features.main(matcher_conf, match_pairs, feature_conf["output"], outputs)

    db_sfm = pycolmap.Reconstruction(sfm_dir)
    db_name_to_id = {img.name: i for i, img in db_sfm.images.items()}

    config = {"estimation": {"ransac": {"max_error": 3}}}
    localizer = QueryLocalizer(db_sfm, config)

    queries = getColmapCameras(q_db_pairs, q_intrinsics)

    # features = Path("datasets/naverlabs/HD_1F/feats-superpoint-nv-n4096-r1024.h5")
    # matches = Path("datasets/naverlabs/HD_1F/feats-superpoint-nv-n4096-r1024_matches-superglue-30_pairs-gt.h5")

    cam_from_world = {}
    logs = { "features": features, "matches": matches, "loc": {}, }

    covisibility_clustering = True
    prepend_camera_name = False

    num_matches = {}

    print("Starting localization...")
    
    # for qname, qcam in tqdm(queries):
    for qname in q_db_pairs.keys():
        
        print(qname)

        qcam = getColmapCamera(qname, q_intrinsics)
        db_ids = [ db_name_to_id[n] for n in q_db_pairs[qname] if n in db_name_to_id ]
        
        if len(db_ids) == 0:
            continue

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, db_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                    localizer, qname, qcam, cluster_ids, features, matches
                )
                if ret is not None and ret["num_inliers"] > best_inliers:
                    best_cluster = i
                    best_inliers = ret["num_inliers"]
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]["PnP_ret"]
                cam_from_world[qname] = ret["cam_from_world"]
            logs["loc"][qname] = {
                "db": db_ids,
                "best_cluster": best_cluster,
                "log_clusters": logs_clusters,
                "covisibility_clustering": covisibility_clustering,
            }

            print("num_inliers:", ret["num_inliers"])
            print("num_matches:", logs_clusters[best_cluster]["num_matches"])
            num_matches[qname] = [logs_clusters[best_cluster]["num_matches"], ret["num_inliers"]]
        else:
            ret, log = pose_from_cluster(
                localizer, qname, qcam, db_ids, features, matches
            )
            if ret is None and log is None:
                continue

            num_matches[qname] = [log["num_matches"], ret["num_inliers"]]

            if ret is not None:
                cam_from_world[qname] = ret["cam_from_world"]
            else:
                closest = db_sfm.images[db_ids[0]]
                cam_from_world[qname] = closest.cam_from_world
                print("Closest image is used", qname)
            log["covisibility_clustering"] = covisibility_clustering
            logs["loc"][qname] = log

    print(f"Localized {len(cam_from_world)} / {len(queries)} images.")
    print(f"Writing poses to {results}...")

    write_results(cam_from_world, num_matches, results, prepend_camera_name)
    write_logs(logs, results)

    print("Done!")

if __name__ == "__main__":

    main()

