# A single object detection from the perception pipeline.

string id                     # stable identifier within a scan cycle, e.g. "blue_mug_0"
string class_name             # YOLO class
string colour_guess           # heuristic, may be empty
string shape_guess            # heuristic, may be empty
float32 confidence

# 2D bounding box in the source image
uint32 bbox_x
uint32 bbox_y
uint32 bbox_w
uint32 bbox_h

# 3D pose in base_link (centroid of the segmented cloud)
geometry_msgs/PoseStamped pose
geometry_msgs/Vector3 size_m  # AABB extents, optional (zeros if unknown)
uint32 num_points             # points in the segmented cloud
