# This branch is an archive for anyone still using ROS1, but is no longer under development. ROS1 issues will also not be answered.
# NerfBridge
![](indoor.gif)

For a complete video see [https://youtu.be/EH0SLn-RcDg](https://youtu.be/EH0SLn-RcDg).

## Introduction
This package implements a bridge between the [Robot Operating System](https://www.ros.org/) (ROS), and the excellent [Nerfstudio](https://docs.nerf.studio/en/latest/) package. Our goal with this package, the NerfBridge Bridge, is to provide a minimal and flexible starting point for robotics researchers to explore possible applications of neural implicit representations.  

In our experience, when it comes to software in robotics, solutions are rarely one size fits all. To that end we cannot provide meticulous installation and implementation instructions that we will be sure will work for every robotics platform. Rather we will try to outline the core components that you will need to get ROS working with Nerfstudio, and hopefully that will get you started on the road to creating some cool NeRFs with your robot.

The core functionality of NerfBridge is fairly simple. At runtime the user provides some basic information about the camera sensor, the name of a ROS Topic that publishes images, and the name of a ROS Topic that publishes a camera pose that corresponds to each image. Using this information NerfBridge starts an instance of Nerfstudio, initializes a ROS Node that listens to the image and pose topics, and pre-allocates two PyTorch Tensors of fixed size (one for the images and one for the corresponding poses). As training commences, images and poses that are received by the NerfBridge node are copied into the pre-allocated data tensors, and in turn pixels are sampled from these data tensors and used to create a NeRF with Nerfstudio. This process continues until the limit of the pre-allocated tensors is reached at which point the NerfBridge stops copying in new images, and training proceeds on the fixed data until completion.

## Requirements
- A Linux machine (tested with Ubuntu 20.04)
	- This should also have a CUDA capable GPU, and fairly strong CPU.
- ROS Noetic installed on your linux machine
- A camera that is compatible with ROS Noetic
- Some means by which to estimate pose of the camera (SLAM, motion capture system, etc)

## Installation  
The first step to getting the NerfBridge working is to install Nerfstudio using the [directions](https://docs.nerf.studio/en/latest/quickstart/installation.html) their documentation. We provide Nerfstudio as a submodule (don't forget to [initialize the submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) after cloning or forking this repo) to this repository so that the version that we have tested this repository with is specified. 

After Nerfstudio and it's dependencies are installed, the only remaining dependency should be ``rospkg``, which can be easily installed using ``pip install rospkg``.

These instructions assume that ROS is already installed on the machine that you will be training on, and that the appropriate ROS packages to provide a stream of color images and poses from your camera are installed and working. For details on the packages that we use to provide this data see the section below on **Our Setup**.

## Running and Configuring NerfBridge
The design and configuration of NerfBridge is heavily inspired by Nerfstudio, and our recommendation is to become familiar with how that repository works before jumping into training NeRFs online with ROS.

Nerfstudio needs three key sources of data to train a NeRF: (1) color images, (2) camera poses corresponding to the color images, and (3) a camera model matching that of the camera that is being used. NerfBridge expects that (1) and (2) are published to corresponding ROS Image and StampedPose topics, and that the names of these topics as well as (3) are provided in a JSON configuration file when the bridge is launched. A sample NerfBridge configuration JSON is provided in the root of the repository, ``nsros_config_sample.json``. We recommend using the [``camera_calibration``](http://wiki.ros.org/camera_calibration) package to determine the camera model parameters. 

Configuring the functionality of NerfBridge is done through the Nerfstudio configuration system, and information about the various settings can be found through the usual addition of the ``-h`` argument added to the launch script ``ros_train.py``. However, since this returns the configurable settings for both Nerfstudio and NerfBridge we provide a brief outline of the NerfBridge specific settings below.

| Option | Description | Default Setting |
| :----- | :--------- | :-----: | 
| ``msg_timeout`` | Before training starts NerfBridge will wait for a set number of posed images to be published on the topics specified in the configuration JSON. This value measures the time (in seconds) that NerfBridge will wait before timing out, and aborting training. | 60.0 (s) |
| ``num_msgs_to_start`` | Number of messages (images and poses) that must have been successfully streamed to Nerfstudio before training will start. | 3 |
| ``draw_training_images`` | Enables an experimental functionality for dynamically updating the camera poses that are visualized in the Nerfstudio Viewer. Right now this mostly doesn't work and will sometimes cause nothing to render in the viewer. | False |
| ``pipeline.datamanager.publish_training_posearray`` | Publish the poses of images that were added to the training dataset as a PoseArray so that they can be visualized in RViz. This essentially circumvents the problems with dynamically updating the camera poses in the Nerfstudio Viewer. | True |
| ``pipeline.datamanager.data_update_freq`` | Frequency, in Hz, that images are added to the training set (allows for sub-sampling of the ROS stream). | 5 (Hz) |
| ``pipeline.datamanager.num_training_images`` | The final size of the training dataset. | 500 |

To launch the NerfBridge (which also starts Nerfstudio) use the command command below.
```
python ros_train.py ros_nerfacto --data /path/to/config.json [OPTIONS]
```
After initializing the Nerfstudio, NerfBridge will show a prompt that it is waiting to receive the appropriate number of images before training starts. When that goal has been reached another prompt will indicate the beginning of training, and then its off to the races!

To set the options above replace ``[OPTIONS]`` with the option name and value. For example:
```
python ros_train.py ros_nerfacto --data /path/to/config.json --pipeline.datamanager.data_update_freq 1.0
```
will set the data update frequency to 1 Hz.

## Our Testing Setup
The following is a description of the testing setup that we at the Stanford Multi-robot Systems Lab have been using to train NeRFs online with NerfBridge.

### Camera
We use an **oCam-1CGN-U-T** camera from WithRobot to provide our stream of images. This camera has quite a few features that make it really nice for working with NeRFs.

- Manual Exposure Control: This helps get consistent lighting in all of the images, and means we can mostly avoid messing with appearance embeddings in our NeRFs.
- Variable Focus: Image sharpness is essential for both the SLAM we use to pose the camera images, and also for achieving well defined geometric features in our NeRFs.
- Global Shutter: We haven't done a lot of tests with rolling shutter cameras, but since we are interested in drone mounted cameras this helps with managing motion blur from vibrations and movement of the camera.
- A Decent ROS Package: The manufacturers of this camera provide a ROS package that is reasonably easy to work with, and is easy to install, [link](https://github.com/withrobot/oCam/tree/master/Software/oCam_ROS_Package/ocam).

We currently use this camera mounted on a quadrotor that publishes images via WiFi to a ground station where the bulk of the computation takes place (Visual SLAM and NeRF training). 

### SLAM
To pose our images we use ORBSLAM3 because it is easy to get working using the [orb_slam3_ros](https://github.com/thien94/orb_slam3_ros) package, and importantly runs entirely on the CPU. In general the hardware bottleneck in a NeRF training pipeline will be the GPU and so this maximizes the resources available for Nerfstudio. 

### Linux Machine
We run NerfBridge on a fairly powerful workstation with a Ryzen 9 5900X CPU, NVIDIA RTX 3090, and 32 GB of DDR4 RAM. This is by no means a minimum specification, but we recommend using a computer with a graphics card with atleast 8GB of VRAM and a modern CPU. Our machine is running Ubuntu 20.04, and has the full desktop version of ROS Noetic installed. 

### Workflow
Our typical workflow is to first launch our drone into a hover, and launch the oCam node on the drones on-board computer. At the same time on the ground station we launch ORBSLAM3, and wait for a preliminary feature map to populate. Once the map is initially populated we run NerfBridge, and either manually fly the drone or launch it into a predefined trajectory. We monitor the NeRF quality through the Nerfstudio Viewer, and the SLAM status and training PoseArray in RViz. 

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
@article{yu2023nerfbridge,
  title={NerfBridge: Bringing Real-time, Online Neural Radiance Field Training to Robotics},
  author={Yu, Javier and Low, Jun En and Nagami, Keiko and Schwager, Mac},
  journal={arXiv preprint arXiv:2305.09761},
  year={2023}
}
```
