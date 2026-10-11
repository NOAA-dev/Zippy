from setuptools import find_packages, setup

package_name = 'state_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='chirag',
    maintainer_email='chiraggujrathi1313@gmail.com',
    description='2D LiDAR-inertial odometry, IMU dead reckoning and odometry visualization',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'lio_node = state_estimation.lio_node:main',
            'imu_odom = state_estimation.imu_odom:main',
            'odom_viz = state_estimation.odom_viz:main',
            'plot_odometry = state_estimation.plot_odometry:main',
            'lio_synthetic_eval = state_estimation.synthetic_eval:main',
        ],
    },
)
