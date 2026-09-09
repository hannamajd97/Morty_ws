from glob import glob

from setuptools import find_packages, setup

package_name = "communication_llm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/scripts", glob("scripts/*.sh")),
    ],
    install_requires=["setuptools", "openai>=1.68.0", "pydantic", "numpy"],
    zip_safe=True,
    maintainer="Majd Hanna",
    maintainer_email="Majd.Hanna@student.reutlingen-university.de",
    description="LLM-based user communication and robot intent extraction.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "chat_node = communication_llm.chat_node:main",
            "speak_action_server = communication_llm.speak_node:main",
            "speech_input_node = communication_llm.speech_input_node:main",
            "mic_visualizer_node = communication_llm.mic_visualizer_node:main",
        ],
    },
)
