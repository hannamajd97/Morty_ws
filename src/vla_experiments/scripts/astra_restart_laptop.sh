#!/usr/bin/env bash
# Restart the laptop-side Astra helpers (static TF base_link->astra_link and the cloud builder).
# Use after the dock cable dropped or the shuttle was restarted. Does not touch RViz or the robot stack.
S=${TMPDIR:-/tmp}
# The supervisor loop first, so it cannot start the node again under us.
# ALL Astra static transforms, not only base_link->astra_link: the four
# internal-frame publishers outlived the launch on every run (2026-09-29: 32
# duplicates after an afternoon of tuning).
for p in $(pgrep -f "astra_cloud_supervis[e]") $(pgrep -f "launch vla_experiments astra_tf.launch.p[y]") $(pgrep -f "static_transform_publishe[r].*astr[a]_") $(pgrep -f "python3.*astra_cloud_loca[l]"); do kill $p 2>/dev/null; done
sleep 2
source /opt/ros/humble/setup.bash; source ~/morty_ws/install/setup.bash
# The stack's profile (2026-09-29): UDP only with 8 MB buffers; the shuttle's
# topics arrive over multicast discovery like they do for the GUI.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/morty_ws/config/fastdds_udp_only.xml
setsid nohup ros2 launch vla_experiments astra_tf.launch.py "$@" > $S/astra_tf.log 2>&1 < /dev/null & disown
# The cloud builder under its supervisor loop (restarts it when the Astra input stops).
setsid nohup bash "$(dirname "$(readlink -f "$0")")/astra_cloud_supervise.sh" > $S/cloudnode.log 2>&1 < /dev/null & disown
sleep 6
echo "astra tf: $(grep -c "from 'base_link'" $S/astra_tf.log) started; cloud builder: $(pgrep -fc 'python3.*astra_cloud_loca[l]') running"
echo "cloud rate: $(timeout 10 ros2 topic hz /astra/depth/points_local 2>&1 | grep -m1 -oE 'average rate: [0-9.]+' || echo 'none yet (is the shuttle camera running? ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh')')"
