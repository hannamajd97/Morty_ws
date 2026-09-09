"""Setup for the ur3e_fast_observation package."""
from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ur3e_fast_observation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer="Majd Hanna",
    maintainer_email="Majd.Hanna@student.reutlingen-university.de",
    description=(
        'Fast observation-pose motion for the UR3e arm. Teach taught joint '
        'poses and replay them via the FollowJointTrajectory action, '
        'bypassing MoveIt for sub-second observation moves.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teach_node = ur3e_fast_observation.teach_node:main',
            'replay_node = ur3e_fast_observation.replay_node:main',
            'import_srdf = ur3e_fast_observation.import_srdf:main',
            'teach_nbv_views = ur3e_fast_observation.teach_nbv_views:main',
            # Action server that wraps replay_node behind a MoveToNamedPose
            # action (with cancellation + MoveIt collision pre-check).
            # Used by intent_router_node and by the cpp grasp executors.
            'move_to_named_pose_server = ur3e_fast_observation.named_pose_server:main',
        ],
    },
)
