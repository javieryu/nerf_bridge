# NerfBridge
![](indoor.gif)

For a complete video see [https://youtu.be/EH0SLn-RcDg](https://youtu.be/EH0SLn-RcDg).

## Introduction
This package implements a bridge between the [Robot Operating System](https://www.ros.org/) (ROS), and the excellent [Nerfstudio](https://docs.nerf.studio/en/latest/) package. Our goal with this package, the NerfBridge Bridge, is to provide a minimal and flexible starting point for robotics researchers to explore possible applications of neural implicit representations.  

In our experience, when it comes to software in robotics, solutions are rarely one size fits all. To that end we cannot provide meticulous installation and implementation instructions that we will be sure will work for every robotics platform. Rather we will try to outline the core components that you will need to get ROS working with Nerfstudio, and hopefully that will get you started on the road to creating some cool NeRFs with your robot.

The core functionality of NerfBridge is fairly simple. At runtime the user provides some basic information about the camera sensor, the name of a ROS Topic that publishes images, and the name of a ROS Topic that publishes a camera pose that corresponds to each image. Using this information NerfBridge starts an instance of Nerfstudio, initializes a ROS Node that listens to the image and pose topics, and pre-allocates two PyTorch Tensors of fixed size (one for the images and one for the corresponding poses). As training commences, images and poses that are received by the NerfBridge node are copied into the pre-allocated data tensors, and in turn pixels are sampled from these data tensors and used to create a NeRF with Nerfstudio. This process continues until the limit of the pre-allocated tensors is reached at which point the NerfBridge stops copying in new images, and training proceeds on the fixed data until completion.

## Requirements
- A Linux machine (tested with Ubuntu 22.04)
	- This should also have a CUDA capable GPU, and fairly strong CPU.
- ROS2 Humble installed on your linux machine
- A camera that is compatible with ROS2 Humble
- Some means by which to estimate pose of the camera (SLAM, motion capture system, etc)

## Installation  
The first step to getting NerfBridge working is to install **just** the dependencies for Nerfstudio using the [installation guide](https://docs.nerf.studio/en/latest/quickstart/installation.html). Then once the dependencies are installed, install Nerfbridge (and Nerfstudio v0.3.3) using ``pip install -e .`` in the root of this repository.

This will add NerfBridge to the list of available methods for on the Nerfstudio CLI. To test if NerfBridge is being registered by the CLI after installation run ``ns-train -h``, and if installation was successful then you should see ``ros-nerfacto`` in the list of available methods.

These instructions assume that ROS2 is already installed on the machine that you will be running NerfBridge on, and that the appropriate ROS packages to provide a stream of color images and poses from your camera are installed and working. For details on the packages that we use to provide this data see the section below on **Our Setup**.

## Running and Configuring NerfBridge
The design and configuration of NerfBridge is heavily inspired by Nerfstudio, and our recommendation is to become familiar with how that repository works before jumping into training NeRFs online with ROS.

Nerfstudio needs three key sources of data to train a NeRF: (1) color images, (2) camera poses corresponding to the color images, and (3) a camera model matching that of the camera that is being used. NerfBridge expects that (1) and (2) are published to corresponding ROS image and pose topics, and that the names of these topics as well as (3) are provided in a JSON configuration file when the bridge is launched. A sample NerfBridge configuration JSON is provided in the root of the repository, ``nerfbridge_config_sample.json``. We recommend using the [``camera_calibration``](http://wiki.ros.org/camera_calibration) package to determine the camera model parameters. 

Configuring the functionality of NerfBridge is done through the Nerfstudio configuration system, and information about the various settings can be found using the command ``ns-train ros-nerfacto -h``. However, since this returns the configurable settings for both Nerfstudio and NerfBridge we provide a brief outline of the NerfBridge specific settings below.

| Option | Description | Default Setting |
| :----- | :--------- | :-----: | 
| ``msg_timeout`` | Before training starts NerfBridge will wait for a set number of posed images to be published on the topics specified in the configuration JSON. This value measures the time (in seconds) that NerfBridge will wait before timing out, and aborting training. | 60.0 (s) |
| ``num_msgs_to_start`` | Number of messages (images and poses) that must have been successfully streamed to Nerfstudio before training will start. | 3 |
| ``draw_training_images`` | Enables an experimental functionality for dynamically updating the camera poses that are visualized in the Nerfstudio Viewer. Right now this mostly doesn't work and will sometimes cause nothing to render in the viewer. | False |
| ``pipeline.datamanager.data_update_freq`` | Frequency, in Hz, that images are added to the training set (allows for sub-sampling of the ROS stream). | 5 (Hz) |
| ``pipeline.datamanager.num_training_images`` | The final size of the training dataset. | 500 |
| ``pipeline.datamanager.slam_method`` | Used to select the correct coordinate transformations and topic types for the streamed poses with options ``[cuvslam, orbslam3]``. **Note:** Currently only tested on ROS2 with the ``cuvslam`` option while using the ``/visual_slam/tracking/odometry`` topic from [ISAAC VSLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam). | ``cuvslam`` |
| ``pipeline.datamanager.topic_sync`` | Selects between ``[exact, approx]`` which correspond to the variety of [TimeSynchronizer](http://docs.ros.org/en/lunar/api/message_filters/html/python) . | ``exact`` |
| ``pipeline.datamanager.topic_slop`` | If an approximate time synchronization is used, then this parameters controls the allowable slop, in seconds, between the image and pose topics to consider a match. | 0.05 (s) |

To launch NerfBridge we use the Nerfstudio CLI with the command command below.
```
ns-train ros-nerfacto --data /path/to/config.json [OPTIONS]
```
After initializing the Nerfstudio, NerfBridge will show a prompt that it is waiting to receive the appropriate number of images before training starts. When that goal has been reached another prompt will indicate the beginning of training, and then its off to the races!

To set the options above replace ``[OPTIONS]`` with the option name and value. For example:
```
ns-train ros-nerfacto --data /path/to/config.json --pipeline.datamanager.data_update_freq 1.0
```
will set the data update frequency to 1 Hz.

## NerfBridge Tutorial on a sample ROS Bag
The script ``scripts/download_data.py`` can be used to download one a sample ROS Bag for testing that a NerfBridge installation is working. The download script is interactive, and can be run with the following commands from the root of the NerfBridge repo:
```
cd scripts
python download_data.py --path /desired/download/location
```
where path specifies the directory to which you would like to download the sample ROS bag. Leaving out the ``--path`` argument will download the ROS Bag to a default location inside of the NerfBridge repo. For the following tutorial use the download script to download the ``desk`` ROS bag.

1. Open two terminals, and in one of them (Terminal 1) execute the following commands to prepare a ROS bag to play.
```
cd /path/to/desk/rosbag

ros2 bag play desk --start-paused
```
This prepares the ROS bag for playing, but does not automatically start it (return to this terminal in Step 3).

2. In the second terminal (Terminal 2), start the ROS bag with the following commands.
```
cd /path/to/nerfbridge/repo

conda activate nerfbridge 
# replacing nerfbridge with the appropriate name of your own conda env from the install process
# or skip this step if you opted to install without conda

ns-train ros-depth-nerfacto --data configs/desk.json --pipeline.datamanager.use-compressed-rgb True
```
Here the ``--data`` argument specifies the configuration data file (topic names etc), and the ``--pipeline.datamanager...`` argument specifies that NerfBridge should expect CompressedImages from the RGB image topic. You should see a message in green that ``(NerfBridge) Waiting for for image streaming to begin ....``.

3. Return to Terminal 1, and press the Space bar to start the ROS bag. Shortly you should see a green message ``(NerfBridge) Dataloader is successfully streaming images!``, and then the training statistics from NerfStudio should begin printing.

4. Terminal 2 should also have printed a link to the Nerfstudio viewer. Follow the link and use the viewer to render views of the NeRF while it trains! Sometimes the viewer can have some latency so if nothing renders the wait a few seconds for it to populate.

If you get to Step 4, and you see a desk with a monitor and laptop appear in the NeRF then your NerfBridge installation is working correctly! Any errors along the way mean that likely your installation is broken. 

## Our Setup
The following is a description of the setup that we at the Stanford Multi-robot Systems Lab have been using to train NeRFs online with NerfBridge from images captured by a camera mounted to a quadrotor.

### Camera
We use a **Intel Realsense D455** camera to provide our stream of images. This camera has quite a few features that make it really nice for working with NeRFs.

  - Integrated IMU: Vision only SLAM tends to be highly susceptible to visual artifacts and lighting conditions requiring more "babysitting". Nicely, the D455 has an integrated IMU which can be used in conjunction with the camera to provide much more reliable pose estimates.
  - Global Shutter: We haven't done a lot of tests with rolling shutter cameras, but since we are interested in drone mounted cameras this helps with managing motion blur from vibrations and movement of the camera.
  - A Great ROS Package: The manufacturers of this camera provide a ROS package that is reasonably easy to work with, and is easy to install, [Intel Realsense ROS](https://github.com/IntelRealSense/realsense-ros).
  - Widely Used: RealSense cameras are widely used throughout the robotics industry which means there are a lot of already available resources for troubleshooting and calibration.

We currently use this camera mounted on a quadrotor that publishes images via WiFi to a ground station where the bulk of the computation takes place (Visual SLAM and NeRF training). 

**Alternatives:** RealSense cameras are expensive, and require more powerful computers to run. For a more economical choice see this camera from Arducam/UCTronics [OV9782](https://www.uctronics.com/arducam-global-shutter-color-usb-1mp-ov9782-uvc-webcam-module.html). This is a bare-bones USB 2.0 camera which can be used in conjunction with the [usb_cam](https://index.ros.org/r/usb_cam/) ROS2 Package. It has the same RGB sensor that is used in the D455, but at a fraction of the cost. However, using this camera will require a different SLAM algorithm than the one that is used in our Testing Setup.

### SLAM
To pose our images we use the [ISAAC ROS Visual SLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) package from NVIDIA. This choice is both out of necessity, and performance. At time of writing, there are few existing and well documented Visual(-Inertial) Odometry packages available for ROS2. Our choice of ISAAC VSLAM is mainly motivated by a few reasons.

  - Relatively easy to install, and has works out of the box with the D455.
  - In our experience, ISAAC VSLAM provides relatively robust performance under a variety of lighting conditions when used with a D455.
  - Directly leverages the compute capability of our on-board computer (see section below for more details).

### Training Machine and On-board Computer
Our current computing setup is composed of two machines a training computer and an on-board computer which are connected via WiFi. The training computer is used to run NerfBridge, and is a powerful workstation with a Ryzen 9 5900X CPU, NVIDIA RTX 3090, and 32 GB of DDR4 RAM. The on-board computer is an NVIDIA Jetson Orin Nano DevKit directly mounted on a custom quadrotor, and is used to run the D455 and SLAM algorithm. At runtime, the on-board camera and training computer communicate over a local wireless network.

Alternatively, everything can run on a single machine with a camera, where the machine runs both the SLAM algorithm and the NerfBridge training. Due to compute requirements this setup will likely not be very "mobile", but can be a good way to verify that everything is running smoothly before testing on robot hardware.

### Drone Workflow
In our typical workflow, we deploy the drone and start the RealSense and SLAM on-board. Then, once the drone is in a steady hover we start NerfBridge on the training machine, and begin the mapping flight orienting the camera towards areas of interest.

## Acknowledgements
NerfBridge is entirely enabled by the first-class work of the [Nerfstudio Development Team and community](https://github.com/nerfstudio-project/nerfstudio/#contributors).

## Citation
In case anyone does use the NerfBridge as a starting point for any research please cite both the Nerfstudio and this repository.

```
# --------------------------- Nerfstudio -----------------------
@article{nerfstudio,
    author = {Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi,
            Brent and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
            Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa, Angjoo},
    title = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
    journal = {arXiv preprint arXiv:2302.04264},
    year = {2023},
}


# --------------------------- NerfBridge ---------------------
@misc{nerfbridge,
    author = {Yu, Javier and Schwager, Mac},
    title = {NerfBridge},
    url = {https://github.com/javieryu/nerf_bridge}
    year = {2023},

}
```
