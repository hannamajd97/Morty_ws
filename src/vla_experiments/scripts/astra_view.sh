#!/usr/bin/env bash
# Laptop side: static TF for the Astra, local cloud builder, and RViz with both cameras.
# Run after the robot stack (for base_link) and after start_astra_on_shuttle.sh on the shuttle.
S=${TMPDIR:-/tmp}
source /opt/ros/humble/setup.bash; source ~/morty_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/morty_ws/config/fastdds_shuttle_peer.xml
pgrep -f "astra_static_t[f]" >/dev/null || { setsid nohup ros2 launch vla_experiments astra_tf.launch.py "$@" > $S/astra_tf.log 2>&1 < /dev/null & disown; }
EXE=$(printf 'astra_cloud_loca%s' l)
pgrep -f "python3.*astra_cloud_loca[l]" >/dev/null || { setsid nohup ros2 run vla_experiments $EXE > $S/cloudnode.log 2>&1 < /dev/null & disown; }
CFG=$(ros2 pkg prefix vla_experiments)/share/vla_experiments/rviz/astra_vs_realsense.rviz
env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/.mutter-Xwaylandauth.AMR5U3} XDG_RUNTIME_DIR=/run/user/1000 QT_QPA_PLATFORM=xcb \
  ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE \
  bash -c "source /opt/ros/humble/setup.bash; source $HOME/morty_ws/install/setup.bash; exec rviz2 -d $CFG"
