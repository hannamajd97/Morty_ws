#!/usr/bin/env python3
"""Clickable debug-image viewer (PyQt5).

Added 2026-08-06, rewritten from cv2 to Qt the same evening: on this machine's
Wayland session, OpenCV's HighGUI windows either never display or hang the
process outright (SYSTEM_STATE_LOG #21/#27/#28/#33), while Qt applications
(rqt_image_view, RViz) display reliably. So this viewer uses Qt — the toolkit
proven to work here.

What it does:
  * subscribes to the annotated debug image (param ``image_topic``,
    default ``/yolo/debug_image``),
  * shows it live in a resizable window,
  * every left-click publishes the click as ``geometry_msgs/Point`` in
    ORIGINAL image pixel coordinates on ``<image_topic>_mouse_left`` — the
    topic ``yolo_live_select_grasp`` already hit-tests against the detection
    boxes for click-to-select.

Click a box → that object is selected. Esc/q or closing the window quits.
"""
from __future__ import annotations

import os
import signal
import sys

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from PyQt5 import QtCore, QtGui, QtWidgets

WINDOW_TITLE = "Morty — click an object to select it"


class _ImageLabel(QtWidgets.QLabel):
    """QLabel that reports left-clicks in ORIGINAL image pixel coords."""

    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self._img_w = 0
        self._img_h = 0

    def set_frame(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        self._img_w, self._img_h = w, h
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(
            pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:  # noqa: N802
        if ev.button() != QtCore.Qt.LeftButton or not self.pixmap() or self._img_w == 0:
            return
        # The pixmap is aspect-fit centred inside the label: undo the offset
        # and the scale to recover original image pixel coordinates.
        pm = self.pixmap()
        off_x = (self.width() - pm.width()) / 2.0
        off_y = (self.height() - pm.height()) / 2.0
        px = ev.pos().x() - off_x
        py = ev.pos().y() - off_y
        if px < 0 or py < 0 or px > pm.width() or py > pm.height():
            return  # clicked the letterbox border, not the image
        sx = self._img_w / float(pm.width())
        sy = self._img_h / float(pm.height())
        self.clicked.emit(px * sx, py * sy)


class ClickSelectViewer(QtWidgets.QMainWindow):
    # The executor launch fired by the EXECUTE button — Majd's "latest" variant
    # from the instructions, with slow scaling and one_shot so a button press
    # runs exactly one pick attempt. (2026-08-06)
    EXECUTOR_CMD = [
        "ros2", "launch", "ur3e_gripper_3d_moveit_commander",
        "grasp_executor_lin.launch.py",
        "best_pose:=true",
        "use_candidate_pose_array:=true",
        "max_candidate_tries:=10",
        "candidate_try_all_batches:=true",
        "override_gripper_pulse_from_executor:=true",
        # Was 0.5/1.5 — it OVERRODE the adapter and would have fought the
        # new 0.20/1.00 defaults on this launch path (Majd 2026-08-28).
        "executor_pulse_hold_s_open:=0.20",
        "executor_pulse_hold_s_close:=1.00",
        "velocity_scaling:=0.1",
        "acceleration_scaling:=0.1",
        "one_shot:=true",
        "execute:=true",
    ]

    def __init__(self, node: Node) -> None:
        super().__init__()
        self._node = node
        self._bridge = CvBridge()
        self._latest: np.ndarray | None = None
        self._exec_proc: "subprocess.Popen | None" = None

        node.declare_parameter("image_topic", "/yolo/debug_image")
        topic = str(node.get_parameter("image_topic").value)

        self._sub = node.create_subscription(Image, topic, self._on_image, 5)
        self._click_pub = node.create_publisher(Point, topic + "_mouse_left", 10)
        # Tells the grasp viewer to paint the executing grasp ORANGE.
        self._exec_pub = node.create_publisher(Bool, "/morty/executor_active", 10)

        self._label = _ImageLabel()
        self._label.clicked.connect(self._on_click)

        # EXECUTE button: spawns the grasp executor launch; becomes STOP while
        # it runs (SIGINT to the whole launch group). The pendant E-stop is
        # still THE emergency stop — this only cancels the software side.
        self._exec_btn = QtWidgets.QPushButton("EXECUTE GRASP  (robot will move!)")
        self._exec_btn.setStyleSheet(
            "QPushButton{background:#1c7c2c;color:white;font-weight:bold;padding:10px;}"
        )
        self._exec_btn.clicked.connect(self._on_exec_clicked)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._label, stretch=1)
        lay.addWidget(self._exec_btn)
        self.setCentralWidget(central)
        self.setWindowTitle(WINDOW_TITLE)
        # Compact default (stream is 640x480 anyway); freely resizable.
        self.resize(760, 620)
        self.statusBar().showMessage(
            f"showing {topic} — left-click a detection box to select that object"
        )

        # Pump ROS from the Qt event loop; repaint when a new frame arrived.
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(20)

        node.get_logger().info(
            f"Viewer up (Qt): showing '{topic}', clicks -> '{topic}_mouse_left'."
        )

    # -- ROS side ----------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        try:
            self._latest = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self._node.get_logger().warn(f"Could not convert image: {exc}")

    def _tick(self) -> None:
        rclpy.spin_once(self._node, timeout_sec=0.0)
        if self._latest is not None:
            self._label.set_frame(self._latest)
            self._latest = None
        # Track the executor subprocess state.
        if self._exec_proc is not None and self._exec_proc.poll() is not None:
            rc = self._exec_proc.returncode
            self._exec_proc = None
            self._exec_pub.publish(Bool(data=False))
            self._exec_btn.setText("EXECUTE GRASP  (robot will move!)")
            self._exec_btn.setStyleSheet(
                "QPushButton{background:#1c7c2c;color:white;font-weight:bold;padding:10px;}"
            )
            self.statusBar().showMessage(f"executor finished (exit code {rc})")
            self._node.get_logger().info(f"Executor launch finished, exit code {rc}.")

    def _on_exec_clicked(self) -> None:
        import os as _os
        import subprocess

        if self._exec_proc is None:
            self._node.get_logger().info(
                "EXECUTE pressed — launching grasp_executor_lin (velocity 10%, one shot)."
            )
            self._exec_proc = subprocess.Popen(
                self.EXECUTOR_CMD, start_new_session=True
            )
            self._exec_pub.publish(Bool(data=True))
            self._exec_btn.setText("EXECUTING…  click to STOP")
            self._exec_btn.setStyleSheet(
                "QPushButton{background:#b02020;color:white;font-weight:bold;padding:10px;}"
            )
            self.statusBar().showMessage("executor running — robot may move")
        else:
            self._node.get_logger().warn("STOP pressed — interrupting executor launch.")
            try:
                _os.killpg(_os.getpgid(self._exec_proc.pid), signal.SIGINT)
            except Exception:
                self._exec_proc.terminate()
            self.statusBar().showMessage("stop requested — waiting for executor to exit")

    def _on_click(self, x: float, y: float) -> None:
        pt = Point()
        pt.x, pt.y, pt.z = float(x), float(y), 0.0
        self._click_pub.publish(pt)
        self._node.get_logger().info(f"Click at image px ({x:.0f},{y:.0f}) published.")

    # -- Qt side -----------------------------------------------------------
    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:  # noqa: N802
        if ev.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.close()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:  # noqa: N802
        # Never leave a robot executor running headless after the window dies.
        if self._exec_proc is not None and self._exec_proc.poll() is None:
            import os as _os

            try:
                _os.killpg(_os.getpgid(self._exec_proc.pid), signal.SIGINT)
            except Exception:
                self._exec_proc.terminate()
        super().closeEvent(ev)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = rclpy.create_node("click_select_viewer")

    # CRITICAL (2026-08-06): importing cv2 (via cv_bridge above) hijacks
    # QT_QPA_PLATFORM_PLUGIN_PATH to point at opencv's BUNDLED Qt plugins,
    # which do not load on this system ("Could not load the Qt platform
    # plugin 'xcb' in .../cv2/qt/plugins"). This single environment variable
    # is why every cv2 HighGUI window failed or hung all evening. Scrub it so
    # PyQt5 finds the real system Qt plugins.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    app = QtWidgets.QApplication(sys.argv)
    # Let Ctrl-C from a terminal / ros2 launch terminate the Qt loop promptly
    # (Qt otherwise swallows SIGINT until the next GUI event — this is what
    # forced launch to escalate to SIGKILL on the cv2 version).
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    win = ClickSelectViewer(node)
    win.show()
    rc = app.exec_()

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
