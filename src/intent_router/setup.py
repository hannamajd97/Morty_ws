from setuptools import find_packages, setup

package_name = "intent_router"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Majd Hanna",
    maintainer_email="Majd.Hanna@student.reutlingen-university.de",
    description="Route /robot/intent JSON to grasp_pipeline parameters.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "intent_router_node = intent_router.intent_router_node:main",
        ],
    },
)
