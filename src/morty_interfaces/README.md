# morty_interfaces

Message, service and action definitions shared by the Morty packages.

| type | name | used by |
| --- | --- | --- |
| action | `OperateGripper` | voice stack → `gripper_action_server` (`/gripper`) |
| action | `MoveToNamedPose` | voice stack and GUI → `move_to_named_pose_server` (`/move_to_named_pose`) |
| action | `Speak` | anything that talks → `speak_action_server` (`/speak`) |
| action | `ExecuteGrasp`, `Pick`, `Place`, `GoHome`, `MoveToPose`, `ScanScene`, `DetectObjects`, `GenerateGrasps`, `SelectObject`, `ListenForChoice` | the action-based interface of the earlier executor variants; kept for compatibility |
| msg | `Detection`, `Grasp` | detection and grasp records |
| srv | `GetSnapshot`, `Stop` | snapshot of the scene; stop |

The live pick path between the pipeline, the executor and the GUI uses plain
ROS message types on topics (see the package READMEs); these definitions
serve the voice stack's actions.
