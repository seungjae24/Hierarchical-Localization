from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_gt_poses,
)

images = Path("/media/seungjae/T7/NaverLabs/Dataset/naverlabs_dataset/HyundaiDepartmentStore/1F/undistorted_images")

outputs = Path("datasets/naverlabs/HD_1F/")
sfm_pairs = outputs / "pairs-gt.txt"
sfm_dir = outputs / "sfm_superpoint+superglue_indoor_30"

pose_path = Path("/media/seungjae/T7/NaverLabs/Dataset/naverlabs_dataset/HyundaiDepartmentStore/1F/release/mapping_with_lidar/sensors/trajectories.txt")

# retrieval_conf  = extract_features.confs["netvlad"]
feature_conf    = extract_features.confs["superpoint_nv"]
matcher_conf    = match_features.confs["superglue"]

# retrieval_path  = extract_features.main(retrieval_conf, images, outputs)
# pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
pairs_from_gt_poses.main(pose_path, sfm_pairs, num_matched=30)

feature_path    = extract_features.main(feature_conf, images, outputs)
match_path      = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs)

# triangulation and BA
# model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, skip_geometric_verification=True)

# triangulation only
