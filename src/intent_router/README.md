# intent_router

Turns the language layer's intents into robot actions. One node,
`intent_router_node`, started by the voice launch.

## In and out

| direction | topic / action | meaning |
| --- | --- | --- |
| in | `/robot/intent` (JSON) | from `chat_node`: pick object X, stop, go home, open/close gripper, drop left/right, look … |
| in | `/yolo/all_detections`, `/joint_states`, `/io_and_status_controller/robot_program_running` | scene, liveness and pendant state used to check an intent is possible |
| out | `/yolo/select_detection`, `/yolo/select_index` | starts a pick by selecting the named object in the pipeline |
| out | `/grasp_pipeline/abort` | stop |
| out | `/grasp_executor_lin_llm` parameters (`/grasp_executor_lin_llm/set_parameters`) | destination and release choices during a pick (`drop_target_name`, waits) |
| out | `/gripper` (action `OperateGripper`), `/move_to_named_pose` (action) | gripper and named-pose moves without a pick |
| out | `/speak` (action) | narration of what is happening |
| out | `/robot/held_object`, `/robot/current_pose` | what the robot holds and where it is, so the chat can answer state questions |

It also listens to `/graspgen/selection_cycle_id` and the executor's
`cycle_done` to narrate the lifecycle, and to `/executor/slip_event` for
"slip detected, slowing down" / "grip restored".

## Parameters worth knowing

`fuzzy_cutoff` (label matching), `scene_max_age_s`, `joint_states_max_age_s`
(refuse a pick when the scene or the robot is stale), `queue_pick_timeout_s`,
`narrate_lifecycle`, `speak_feedback`, `home_pose_name`, `ready_pose_name`,
`gripper_force_n`, `enable_vitamin_mode` (a different close-pulse pattern for
the vitamin tube; off by default; unrelated to slip handling).
