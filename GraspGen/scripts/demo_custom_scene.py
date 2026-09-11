import argparse
import json
import os
import numpy as np
import torch
import meshcat

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer, get_color_from_score, visualize_grasp
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize grasps on a custom colored scene")
    parser.add_argument("--sample_data_dir", type=str, required=True,
                        help="Directory containing scene.json")
    parser.add_argument("--gripper_config", type=str, required=True,
                        help="Path to gripper config YAML file")
    parser.add_argument("--grasp_threshold", type=float, default=0.4,
                        help="Confidence threshold for showing grasps")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    json_file = os.path.join(args.sample_data_dir, "scene.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"scene.json not found in {args.sample_data_dir}")

    # Load gripper config and model
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # Create MeshCat visualizer
    vis = create_visualizer()
    vis.delete()

    # Load scene data
    with open(json_file, "r") as f:
        data = json.load(f)

    obj_pc = np.array(data["object_info"]["pc"], dtype=np.float32)
    obj_pc_color = np.array(data["object_info"]["pc_color"], dtype=np.float32).reshape(-1, 3)

    # Debug color values
    print("\n[DEBUG] Checking loaded point cloud colors...")
    print("Shape:", obj_pc_color.shape)
    print("First 10 colors (raw):")
    print(obj_pc_color[:10])
    print("Min color value:", obj_pc_color.min(), "Max color value:", obj_pc_color.max())

    # Ensure point count matches
    assert obj_pc.shape[0] == obj_pc_color.shape[0], \
        f"Point/color count mismatch: {obj_pc.shape[0]} vs {obj_pc_color.shape[0]}"

    if len(obj_pc) == 0:
        print("Error: Captured point cloud is empty.")
        exit()

    # Downsample if too many points
    NUM_POINTS = 20000
    if len(obj_pc) > NUM_POINTS:
        print(f"--- Downsampling from {len(obj_pc)} to {NUM_POINTS} points.")
        indices = np.random.choice(len(obj_pc), NUM_POINTS, replace=False)
        obj_pc, obj_pc_color = obj_pc[indices], obj_pc_color[indices]

    # Convert colors to 0–1 range if needed
    if obj_pc_color.max() > 1.0:
        obj_pc_color = obj_pc_color / 255.0

    # Plot the colored point cloud in MeshCat
    pc_geom = meshcat.geometry.PointCloud(
        position=obj_pc.T,   # shape (3, N)
        color=obj_pc_color.T # shape (3, N)
    )
    vis["scene"]["pointcloud"].set_object(pc_geom)

    # Run grasp inference
    grasps, conf = GraspGenSampler.run_inference(
        obj_pc, grasp_sampler, grasp_threshold=args.grasp_threshold
    )

    if len(grasps) > 0:
        print(f"Found {len(grasps)} grasps with confidence > {args.grasp_threshold}")
        for i, (g, c) in enumerate(zip(grasps.cpu().numpy(), conf.cpu().numpy())):
            color = get_color_from_score(c, use_255_scale=False)
            visualize_grasp(vis, f"grasp_{i}", g, color=color,
                            gripper_name=grasp_cfg.data.gripper_name)
        print("\nSUCCESS: Grasps are visualized in MeshCat.")
    else:
        print(f"No grasps found. Try lowering threshold or improving scene.")

    input("Press Enter to exit.")

