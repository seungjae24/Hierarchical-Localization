from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_gt_poses,
    triangulation,
)

from hloc.utils.read_write_model import (
    Camera,
    Image,
    qvec2rotmat,
    rotmat2qvec,
    write_model,
)

import numpy as np
import pycolmap
import cv2

from common_util import configs

def cam_from_calibration_file(id_, path):
    """Create a COLMAP camera from an MLAD calibration file."""
    with open(path, "r") as f:
        data = f.readlines()
    model, fx, fy, cx, cy = data[0].split()[:5]
    width, height = data[1].split()
    assert model == "Pinhole"
    model_name = "PINHOLE"
    params = [float(i) for i in [fx, fy, cx, cy]]
    camera = Camera(
        id=id_, model=model_name, width=int(width), height=int(height), params=params
    )
    return camera

def parse_poses(path):

    poses = {}
    with open(path) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            elif "lidar" in line.split(", ")[1]:
                continue
            else:
                timestamp = line.split(", ")[0]
                device_id = line.split(", ")[1].split("_")[0]
                id = device_id + "_" + timestamp #line.split(",")[1].split("_")[0] + "_" + line.split(",")[0]
                qvec = [float(x) for x in line.split(",")[2:6]]
                tvec = [float(x) for x in line.split(",")[6:9]]
                
                poses[id] = {"qvec": qvec, "tvec": tvec}

    return poses

def read_intrinsics(intrinsic_path):
    intrinsics = {}
    with open(intrinsic_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            elif "lidar" in line:
                continue
            elif "depth" in line:
                continue
            else:
                device_id = line.split(",")[0].split("_")[0]
                model = line.split(",")[3][1:]
                width = line.split(",")[4][1:]
                height = line.split(",")[5][1:]
                params = [x[1:] for x in line.split(",")[6:]]
                params = [float(x) for x in params]

                intrinsics[device_id] = {"model": model, "width": width, "height": height, "params": params}    
    return intrinsics

def build_base_colmap_model(data_path, sfm_dir):
    """Build a COLMAP model with images and cameras only."""

    file_path       = data_path + "/mapping_with_lidar/sensors/records_camera.txt"
    pose_path       = data_path + "/mapping_with_lidar/sensors/trajectories.txt"
    intrinsic_path  = data_path + "/mapping_with_lidar/sensors/sensors.txt"
    
    poses           = parse_poses(pose_path)
    intrinsics      = read_intrinsics(intrinsic_path)

    image_files = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            elif "lidar" in line:
                continue
            elif "depth" in line:
                continue
            else:
                line = line.strip()
                timestamp = line.split(", ")[0]
                device_id = line.split(", ")[1].split("_")[0]
                image_name = device_id + "_" + timestamp
                image_file = line.split(", ")[2]

                image_files[image_file] = image_name

    cameras = {}
    for device_id in intrinsics.keys():
        intrinsic = intrinsics[device_id]

        camera = Camera(
            id=int(device_id),
            model=intrinsic["model"],
            width=int(intrinsic["width"]),
            height=int(intrinsic["height"]),
            params=intrinsic["params"],
        )
        cameras[device_id] = camera

    images = {}
    id_ = 0
    for image_file in image_files.keys():

        im_name = image_files[image_file]
        pose = poses[im_name]
        q = np.array(pose["qvec"])
        t = np.array(pose["tvec"])
        device_id = im_name.split("_")[0]

        image = Image(
            id=id_,
            qvec=q,
            tvec=t,
            camera_id=int(device_id),
            name=image_file,
            xys=np.zeros((0, 2), float),
            point3D_ids=np.full(0, -1, int),
        )
        images[im_name] = image
        id_ += 1

    sfm_dir.mkdir(exist_ok=True, parents=True)
    write_model(cameras, images, {}, path=str(sfm_dir), ext=".bin")

def main():

    seq             = "HD_1F"
    data_path       = configs[seq]["data_path"]

    db_image_path   = data_path + "/mapping_with_lidar/sensors/records_data/"
    db_pose_path    = data_path + "/mapping_with_lidar/sensors/trajectories.txt"
    
    outputs = Path("outputs/naverlabs/" + seq)
    outputs.mkdir(exist_ok=True, parents=True)

    sfm_pairs       = outputs / "pairs-db.txt"
    sfm_dir         = outputs / "sfm_r2d2+NN-ratio"
    sfm_dir_empty   = outputs / "sfm_empty"
    
    sfm_dir.mkdir(exist_ok=True, parents=True)
    sfm_dir_empty.mkdir(exist_ok=True, parents=True)

    # retrieval_conf  = extract_features.confs["netvlad"]
    feature_conf    = extract_features.confs["r2d2"]
    matcher_conf    = match_features.confs["NN-ratio"]

    build_base_colmap_model(data_path, sfm_dir_empty)

    retrieval_conf  = extract_features.confs["netvlad-db"]
    retrieval_path  = extract_features.main(retrieval_conf, Path(db_image_path), outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=30)
    # pairs_from_gt_poses.main(db_pose_path, sfm_pairs, num_matched=30)

    feature_path    = extract_features.main(feature_conf, Path(db_image_path), outputs)
    match_path      = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs)

    # triangulation and BA
    # model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, skip_geometric_verification=True)

    # triangulation only
    triangulation.main(sfm_dir, sfm_dir_empty, Path(db_image_path), sfm_pairs, feature_path, match_path)


if __name__ == "__main__":
    main()