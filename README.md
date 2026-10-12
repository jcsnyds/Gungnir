# Gungnir

#### In Norse mythology, Gungnir (/ˈɡʌŋ.nɪər/, "the rocker, the swaying one") is the spear of the god Odin. It is known for always hitting the target of the attacker regardless of the attacker's skill.

The goal of this project is to develop a system capable of picking up any object specified by the user with a 6 DOF robot arm
using a depth camera and grasping model. The system will be able to identify the requested item in the frame, compute all possible
ways the object could be grasped, and then pick it up with a physical gripper.

## Getting Started

#### First clone this repository.

```
git clone https://github.com/jcsnyds/Gungnir.git
cd Gungnir
```

#### Create a new virtual environment for this project.

`python -m venv .venv`

#### Activate the virtual environment.

For Windows: `.venv\Scripts\activate.bat`

For Linux: `source .venv/bin/activate`

#### Then run the setup script.

`python setup.py`

Follow any prompts given to install the necessary packages and software.
