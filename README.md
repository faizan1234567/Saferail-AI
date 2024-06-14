# Saferail-AI

Pakistan Railway is the only mode of public transport which provides cheap means of transportation. It carried 52.2 million passengers in 2016 and operates 28 mail, express and passenger trains. The railway carries a daily average of 178, 000 passengers. Despite its cheap mode of transportation, Pakistan railway suffered various accidents. Over the past five years, 537 train accidents took place of which 313 accidents were such that led to loss of life or serious injury. Factorial analysis suggests that 32% of such accidents happened at unmanned level crossing for which the road users were responsible. The conventional solution is to detect objects using an optical camera to prevent level crossing accidents that may fail in low lighting conditions such as nighttime, fog, and severe weather. 

To mitigate this challenge, we combine an infrared imaging camera which captures variations in ambient temperature emitted from objects, highlighting structures of thermal target insensitive to lighting variations.  The fusion of optical and infrared information provides remarkable advantages in all weather, lighting, and challenging conditions. We further uses AI based object detection, depth estimation, and ROI segmentation algorithms to get the classification of the target and its distance from the train if the object is located on the track. And then  uses an MMI display to show the distance and classification of the target to the driver to generate a warning and control action timely to prevent potentail accidents. 

## Installation
The implementation is in progress, we intend to run image fusion and object detection algorithms on NVIDIA's Jetson Orin Nano 8GB. This repository initially implements image fusion and object detection for SafeRail application. Later we will incorporate the depth estimation and ROI segmentaion algorithms with powerful hardware accelerators.

```bash
 git clone https://github.com/faizan1234567/Saferail-AI
 cd Saferail-AI
```

Create  a virtual enviroment
```bash
python3 -m venv saferail
source saferail/bin/activate
```

Now install all the required dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Installation compelete.

Note: as we are deploying it on the jetson orin, please make sure to install pytorch version that matches with your jetpack version,
more information: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

the implementation is in progress




