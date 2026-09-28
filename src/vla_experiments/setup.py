from glob import glob

from setuptools import find_packages, setup

package_name = "vla_experiments"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.xml")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}/rlds_builder/morty_ur3e", glob("rlds_builder/morty_ur3e/*.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Majd Hanna",
    maintainer_email="Majd.Hanna@student.reutlingen-university.de",
    description="Demonstration recording and dataset export for VLA fine-tuning.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "episode_recorder = vla_experiments.episode_recorder:main",
            "vla_episode = vla_experiments.episode_cli:main",
            "vla_dataset_info = vla_experiments.raw_dataset:main",
            "vla_view = vla_experiments.episode_viewer:main",
            "vla_relabel = vla_experiments.relabel:main",
            "convert_to_lerobot = vla_experiments.convert_to_lerobot:main",
            "convert_to_rlds = vla_experiments.convert_to_rlds:main",
            "astra_cloud_local = vla_experiments.astra_cloud_local:main",
            "astra_calibrate = vla_experiments.astra_calibrate:main",
        ],
    },
)
