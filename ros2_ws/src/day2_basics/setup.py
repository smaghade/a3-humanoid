from setuptools import find_packages, setup

package_name = 'day2_basics'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='antpc',
    maintainer_email='skandru@deepedge.ai',
    description='Day 2: ROS 2 publisher and subscriber',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = day2_basics.talker:main',
            'listener = day2_basics.listener:main',
        ],
    },
)
