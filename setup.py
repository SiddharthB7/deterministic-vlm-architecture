from setuptools import find_packages, setup


package_name = "kurat"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["kurat_core", "kurat_core.*", "kurat_io", "kurat_io.*", "kurat_ros", "kurat_ros.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["kurat_ros/launch/kurat.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kurat",
    maintainer_email="user@example.com",
    description="Kurat multimodal assistant ROS 2 package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "kurat_orchestrator_node = kurat_ros.orchestrator_node:main",
        ],
    },
)
