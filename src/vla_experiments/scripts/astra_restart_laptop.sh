#!/usr/bin/env bash
# Restart the laptop-side Astra helpers (static TF base_link->astra_link and the cloud builder).
# Use after the dock cable dropped or the shuttle was restarted. Does not touch RViz or the robot stack.
S=${TMPDIR:-/tmp}
for p in $(pgrep -f "astra_static_t[f]") $(pgrep -f "launch vla_experiments astra_tf.launch.p[y]") $(pgrep -f "python3.*astra_cloud_loca[l]"); do kill $p 2>/dev/null; done
sleep 2
source /opt/ros/humble/setup.bash; source ~/morty_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/morty_ws/config/fastdds_shuttle_peer.xml
setsid nohup ros2 launch vla_experiments astra_tf.launch.py "$@" > $S/astra_tf.log 2>&1 < /dev/null & disown
EXE=$(printf 'astra_cloud_loca%s' l)
setsid nohup ros2 run vla_experiments $EXE > $S/cloudnode.log 2>&1 < /dev/null & disown
sleep 6
echo "astra tf: $(grep -c "from 'base_link'" $S/astra_tf.log) started; cloud builder: $(pgrep -fc 'python3.*astra_cloud_loca[l]') running"
echo "cloud rate: $(timeout 10 ros2 topic hz /astra/depth/points_local 2>&1 | grep -m1 -oE 'average rate: [0-9.]+' || echo 'none yet (is the shuttle camera running? ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh')')"
