# Real-Time-PL-VIO
##  Tightly-Coupled Real Time Monocular Visual–Inertial Odometry Using Point and Line Features

Based on [PL-VIO](https://github.com/HeYijia/PL-VIO), Real-Time-PL-VIO uses [EDlines](https://github.com/CihanTopal/ED_Lib) to extracte line features, and Optical flow to track them, Helmholtz principle to check tracking results. Real-Time-PL-VIO can achieve a real time performance with CPU.

This code runs on **Linux**, and is fully integrated with **ROS**. 

## 1. Prerequisites
1.1 **Ubuntu** and **ROS**
Ubuntu 16.04. ROS Kinetic, [ROS Installation](http://wiki.ros.org/indigo/Installation/Ubuntu)
additional ROS pacakge

```
	sudo apt-get install ros-YOUR_DISTRO-cv-bridge ros-YOUR_DISTRO-tf ros-YOUR_DISTRO-message-filters ros-YOUR_DISTRO-image-transport
```
If you install ROS Kinetic, please update **opencv3** with 
```
    sudo apt-get install ros-kinetic-opencv3
```

1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html), remember to **make install**.

## 2. Build PL-VIO on ROS
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/zhouhaoran-TJU/Real-Time-PL-VIO.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## 3.Performance on EuRoC dataset

### 3.1 Run with EuRoC dataset directly
3.1.1 Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). Although it contains stereo cameras, we only use one camera.

3.1.2 Open three terminals, launch the vins_estimator , rviz and play the bag file respectively. Take MH_05 as example

```c++
    roslaunch plvio_estimator euroc_fix_extrinsic.launch 
    roslaunch plvio_estimator vins_rviz.launch 
    rosbag play YOUR_PATH_TO_DATASET/MH_05_difficult.bag 
```

## 4. Acknowledgements

RTPL-VIO use [PL-VIO](https://github.com/HeYijia/PL-VIO) as base line code.
