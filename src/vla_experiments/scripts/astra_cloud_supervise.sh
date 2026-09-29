#!/usr/bin/env bash
# Keeps the laptop-side Astra cloud builder running (2026-09-29). The node exits
# after input_timeout_s without Astra depth (camera not up yet at boot, shuttle
# rebooted, driver frozen); this loop starts a fresh one 5 s later. While it
# waits it costs a sleeping bash; a restart about once a minute while the
# shuttle camera is down.
EXE=$(printf 'astra_cloud_loca%s' l)
while true; do
    ros2 run vla_experiments "$EXE" "$@"
    sleep 5
done
