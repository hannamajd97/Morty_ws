from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_setup_assistant_launch


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
        "ur3e_with_gripper_3d", package_name="ur3e_gripper_3d_moveit_config"
        )
        .robot_description(file_path="urdf/ur3e_gripper_wrapper.urdf.xacro")
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl",
            load_all=False,
        )
        .to_moveit_configs()
    )
    return generate_setup_assistant_launch(moveit_config)
