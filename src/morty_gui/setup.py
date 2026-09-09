from setuptools import setup

package_name = "morty_gui"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Majd Hanna",
    maintainer_email="Majd.Hanna@student.reutlingen-university.de",
    description="Unified operator + advanced-debug GUI for the Morty grasping pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "morty_gui = morty_gui.app:main",
        ],
    },
)
