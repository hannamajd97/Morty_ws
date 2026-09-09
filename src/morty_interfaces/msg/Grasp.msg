# A single grasp candidate.

geometry_msgs/PoseStamped pose          # gripper pose in base_link
geometry_msgs/PoseStamped pre_grasp_pose
geometry_msgs/PoseStamped lift_pose
float32 width_m                          # required gripper opening
float32 score                            # GraspGen score
float32 score_after_filters              # score after reweighting; equal to `score` if no reweight
string source                            # "graspgen" | "centroid_fallback" | other
