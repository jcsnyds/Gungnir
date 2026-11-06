# Gungnir

#### In Norse mythology, Gungnir (/ˈɡʌŋ.nɪər/, "the rocker, the swaying one") is the spear of the god Odin. It is known for always hitting the target of the attacker regardless of the attacker's skill.

The goal of this project is to develop a system capable of picking up any object specified by the user with a 6 DOF robot arm
using a depth camera and grasping model. The system will be able to identify the requested item in the frame, compute all possible
ways the object could be grasped, and then pick it up with a physical gripper.

## Getting Started

### Important Notes
- **This code is designed to run on two computers**
  - One computer with a GPU runs the Contact-Graspnet Server
  - The other computer (with or without a GPU) runs the Gungnir Client
- **Follow the setup instructions below on** ***Both*** **computers**. 

### First clone this repository.

```
git clone https://github.com/jcsnyds/Gungnir.git
cd Gungnir
```

### Create a new virtual environment for this project.

`python -m venv .venv`

### Activate the virtual environment.

For Windows: `.venv\Scripts\activate.bat`

For Linux: `source .venv/bin/activate`

### Then run the setup script on both computers

`python setup.py`

Follow any prompts given to install the necessary packages and software.

### **On the Server computer**
- Connect the Server computer to the Client computer with an Ethernet cable. Then ssh into the Server and run the following command.

`python ContactGraspnet_Server.py`

This will output the Server's IP address to be used in the next step.

### **On the Client computer**
- Plug in the Intel Realsense L515 to a USB port then run the following command using the IP Address found from the step above.

`python Gungnir_Client.py <gpu-computer-ip>`

## Technical Details

### Division of Labor

- **GPU Enabled Computer (Server)**
  - Get an image, segmentation map, and depth map 
  - Returns the best grasp locations
  - Communicates over HTTP to the Gungnir Client

- **Generic Computer (Client)**
  - Captures images from an Intel RealSense L515 depth camera
  - Performs segmentation on captured images
  - Sends data to the Contact-Graspnet Server
