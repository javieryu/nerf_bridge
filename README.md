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
- ROS2 Humble installed on your Linux machine
- A camera that is compatible with ROS2 Humble
- Some means by which to estimate pose of the camera (SLAM, motion capture system, etc)

## Installation  
The first step to getting NerfBridge working is to install **just** the dependencies for Nerfstudio using the [installation guide](https://docs.nerf.studio/en/latest/quickstart/installation.html). Then once the dependencies are installed, install Nerfbridge (and Nerfstudio v1.0) using ``pip install -e .`` in the root of this repository.

This will add NerfBridge to the list of available methods for on the Nerfstudio CLI. To test if NerfBridge is being registered by the CLI after installation run ``ns-train -h``, and if installation was successful then you should see ``ros-nerfacto`` in the list of available methods.

These instructions assume that ROS2 is already installed on the machine that you will be running NerfBridge on, and that the appropriate ROS packages to provide a stream of color images and poses from your camera are installed and working. For details on the packages that we use to provide this data see the section below on **Our Setup**.

## Installation
Installing NerfBridge is a rather involved process because it requires using Anaconda alongside ROS2. To that end, below is a guide for installing NerfBridge on an x86\_64 Ubuntu 22.04 machine with an NVIDIA GPU.

1. Install ROS2 Humble using the [installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).

2. Install Miniconda (or Anaconda) using the [installation guide](https://docs.anaconda.com/free/miniconda/miniconda-install/).

3. Create a conda environment for NerfBridge. Take note of the optional procedure for completely isolating the conda environment from your machine's base python site packages. For more details see this [StackOverflow post](https://stackoverflow.com/questions/25584276/how-to-disable-site-enable-user-site-for-an-environment) and [PEP-0370](https://peps.python.org/pep-0370/). 

    ```bash
    conda create --name nerfbridge -y python=3.10
    
    # OPTIONAL, but recommended when using 22.04
    conda activate nerfbridge
    conda env config vars set PYTHONNOUSERSITE=1
    conda deactivate
    ```

4. Activate the conda environment and install Nerfstudio dependencies.

    ```bash
    # Activate conda environment, and upgrade pip
    conda activate nerfbridge
    python -m pip install --upgrade pip

    # PyTorch, Torchvision dependency
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

    # CUDA dependency (by far the easiest way to manage cuda versions)
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

    # TinyCUDANN dependency (takes a while!)
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    # GSPLAT dependency (takes a while!)
    # Avoids building at runtime of first splatfacto training.
    pip install git+https://github.com/nerfstudio-project/gsplat.git
    ```

5. Clone, and install NerfBridge.

    ```bash
    # Clone the NerfBridge Repo
    git clone https://github.com/javieryu/nerf_bridge.git

    # Install NerfBridge (and Nerfstudio as a dependency)
    # Make sure your conda env is activated!
    cd nerf_bridge
    pip install -e . 
    ```

Now you should be setup to run NerfBridge. In the next section is a basic tutorial on training your first NeRF using NerfBridge.

## Example using a ROSBag
This example simulates streaming data from a robot by replaying a [ROS2 bag](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and using NerfBridge to train a Nerfacto model on that data. This example is a great way to check that your installation of NerfBridge is working correctly.

These instructions are designed to be run in two different terminal sessions. They will be reffered to as terminals 1 and 2.

1. [Terminal 1] Download the example `desk` rosbag using the provided download script. This script creates a folder `nerf_bridge/rosbags`, and downloads a rosbag to that directory.
    ```bash
    # Activate nerfbirdge conda env
    activate nerfbridge

    # Run download script
    cd nerf_bridge
    python scripts/download_data.py
    ```

2. [Terminal 2] Start the `desk` rosbag paused. The three important ROS topics in this rosbag are an image topic (`/camera/color/image_raw/compressed`), a depth topic (`/camera/aligned_depth_to_color/image_raw`), and a camera pose topic (`/visual_slam/tracking/odometry`). These will be the data streams that will be used to train the Nerfacto model using NerfBridge.

    ```bash
    # NOTE: for this example, Terminal 2 does not need to use the conda env.

    # Play the rosbag, but start it paused.
    cd nerf_bridge/rosbags
    ros2 bag play desk --start-paused
    ```

3. [Terminal 1] Start NerfBridge using the Nerfstudio `ns-train` CLI. Notice, included are some parameters that must be changed from the base NerfBridge configuration to accomodate the scene. The function of these parameters is outlined in the next section. 

    ```bash
    ns-train ros-depth-nerfacto --data configs/desk.json --pipeline.datamanager.use-compressed-rgb True --pipeline.datamanager.dataparser.scene-scale-factor 0.5 --pipeline.datamanager.data-update-freq 8.0
    ```

    After some initialization a message will appear stating that `(NerfBridge) Images recieved: 0`, at this point you should open the Nerfstudio viewer in a browser tab to visualize the Nerfacto model as it trains. Training will start after completing the next step.

5. [Terminal 2] Press the SPACE key to start playing the rosbag. Once the pre-training image buffer is filled (defaults to 10 images) then training should commence, and the usual Nerfstudio print messages will appear in Terminal 1. After a few seconds the Nerfstudio viewer should start to show the recieved images as camera frames, and the Nerfacto model should begin be filled out.

6. After the rosbag in Terminal 2 finishes playing NerfBridge will continue training the Nerfacto model on all of the data that it has recieved, but no new data will be added. You can use CTRL+c to kill NerfBridge after you are done inspecting the Nerfacto model in the viewer.

## Running and Configuring NerfBridge
The design and configuration of NerfBridge is heavily inspired by Nerfstudio, and our recommendation is to become familiar with how that repository works before jumping into your own custom implementation of NerfBridge.

Nerfstudio needs three key sources of data to train a NeRF: (1) color images, (2) camera poses corresponding to the color images, and (3) a camera model matching that of the camera that is being used. NerfBridge expects that (1) and (2) are published to corresponding ROS image and pose topics, and that the names of these topics as well as (3) are provided in a JSON configuration file at launch. A sample NerfBridge configuration JSON is provided in the root of the repository, `nerf_bridge/configs/desk.json` (this is the config used for the example above). We recommend using the [``camera_calibration``](http://wiki.ros.org/camera_calibration) package to determine the camera model parameters. 

At present we support two Nerfstudio architectures out of the box `nerfacto` and `splatfacto`. For each of these architectures we support both RGB only training, and RGBD training. To use any of the methods simply provide the correct JSON configuration file (if using a depth supervised model then also specify the appropriate depth related configurations). The method names provided by NerfBridge are `ros-nerfacto` (Nerfacto RGB), `ros-depth-nerfacto` (Nerfacto RGBD), `ros-splatfacto` (Splatfacto RGB), and `ros-depth-splatfacto` (Splatfacto RGBD).

Configuring the runtime functionality of NerfBridge is done through the Nerfstudio CLI configuration system, and information about the various settings can be found using the command ``ns-train ros-nerfacto -h``. However, since this returns the configurable settings for both Nerfstudio and NerfBridge we provide a brief outline of the NerfBridge specific settings below.

| Option | Description | Default Setting |
| :----- | :--------- | :-----: | 
| `msg-timeout` | Before training starts NerfBridge will wait for a set number of posed images to be published on the topics specified in the configuration JSON. This value measures the time (in seconds) that NerfBridge will wait before timing out, and aborting training. | 300.0 (float, s) |
| `num-msgs-to-start` | Number of messages (images and poses) that must have been successfully streamed to Nerfstudio before training will start. | 10 (int)|
| `pipeline.datamanager.data-update-freq` | Frequency, in Hz, that images are added to the training set (allows for sub-sampling of the ROS stream). | 5.0 (float, Hz) |
| `pipeline.datamanager.num-training-images` | The final size of the training dataset. | 500 (int)|
| `pipeline.datamanager.slam-method` | Used to select the correct coordinate transformations and message types for the streamed poses with options ``[cuvslam, mocap, orbslam3]``. **Note:** Currently only tested on ROS2 with the ``cuvslam`` option while using the ``/visual_slam/tracking/odometry`` topic from [ISAAC VSLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam). | ``cuvslam`` (string)|
| `pipeline.datamanager.topic-sync` | Selects between ``[exact, approx]`` which correspond to the variety of [TimeSynchronizer](http://docs.ros.org/en/lunar/api/message_filters/html/python) used to subscribe to topics. | ``exact`` (string)|
| `pipeline.datamanager.topic-slop` | If an approximate time synchronization is used, then this parameters controls the allowable slop, in seconds, between the image and pose topics to consider a match. | 0.05 (float, s) |
| `pipeline.datamanager.use-compressed-rgb` | Whether the RGB image topic is a CompressedImage topic or not. | False (bool) |
| `pipeline.datamanager.dataparser.scene-scale-factor` | How much to scale the origins by to ensure that the scene fits within [-1, 1] box used for the Nerfacto models. | 1.0 (float) |
| `pipeline.model.depth_seed_pts` | [ONLY for `ros-depth-splatfacto`] Configures how many gaussians to create from each depth image. | 2000 (int) |

To launch NerfBridge we use the Nerfstudio CLI with the command command below.
```
ns-train [METHOD-NAME] --data /path/to/config.json [OPTIONS]
```
After initializing the Nerfstudio, NerfBridge will show a prompt that it is waiting to receive the appropriate number of images before training starts. When that goal has been reached another prompt will indicate the beginning of training, and then its off to the races!

To set the options above replace ``[OPTIONS]`` with the option name and value. For example:
```
ns-train ros-nerfacto --data /path/to/config.json --pipeline.datamanager.data_update_freq 1.0
```
will set the data update frequency to 1 Hz.

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
@article{yu2023nerfbridge,
  title={NerfBridge: Bringing Real-time, Online Neural Radiance Field Training to Robotics},
  author={Yu, Javier and Low, Jun En and Nagami, Keiko and Schwager, Mac},
  journal={arXiv preprint arXiv:2305.09761},
  year={2023}
}
```
