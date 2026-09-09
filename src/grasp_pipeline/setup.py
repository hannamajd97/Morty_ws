from glob import glob

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

package_name = 'grasp_pipeline'

# --- SHIMS: accept/ignore unknown --editable passed by old ament/colcon ---
class DevelopCommand(develop):
    # add a dummy --editable option so setuptools' arg parser doesn't explode
    user_options = develop.user_options + [
        ('editable', None, 'compat flag (ignored)'),
        ('build-directory=', None, 'build directory (ignored)'),
        ('script-dir=', None, 'script directory'),
    ]
    def initialize_options(self):
        super().initialize_options()
        self.editable = True  # harmless default
        self.build_directory = None
        self.script_dir = None

class InstallCommand(install):
    # some stacks pass through options oddly; be liberal:
    user_options = install.user_options + [
        ('editable', None, 'compat flag (ignored)'),
        ('build-directory=', None, 'build directory (ignored)'),
    ]
    def initialize_options(self):
        super().initialize_options()
        self.editable = True
        self.build_directory = None

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    # MediaPipe Tasks hand model (2026-09-03, point_selector 3d mode).
    package_data={package_name: ['models/*.task']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Bash wrapper that lets `ros2 run` execute inside a venv
        ('lib/' + package_name, ['scripts/run_grasp_pipeline_venv']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'open3d',
        'opencv-python',
        'pyzmq',
    ],
    zip_safe=True,
    author='You',
    author_email='you@example.com',
    description='YOLO detection + segmentation + point cloud filtering + GraspGen client',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Core pipeline nodes
            'grasp_selection_node = grasp_pipeline.grasp_selection_node:main',
            'run_grasp_pipeline = grasp_pipeline.run_grasp_pipeline:main',
            'scene_obstacle_publisher = grasp_pipeline.scene_obstacle_publisher:main',
            'pose_to_base = grasp_pipeline.pose_to_base:main',
            'open3d_cloud_viewer = grasp_pipeline.open3d_cloud_viewer:main',
            'click_select_viewer = grasp_pipeline.click_select_viewer:main',
            'grasp_viz_viewer = grasp_pipeline.grasp_viz_viewer:main',
            # Standalone cloud debug tools (live under grasp_pipeline/centroid/)
            'cloud_metrics = grasp_pipeline.centroid.cloud_metrics:main',
            'cloud_centroid_pub = grasp_pipeline.centroid.cloud_centroid_pub:main',
            'cloud_saver = grasp_pipeline.centroid.cloud_saver:main',
        ],
    },
    cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,
    },
)
