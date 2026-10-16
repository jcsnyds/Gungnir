import subprocess as s

s.run(["python", "IntelRealsense_Driver/setup.py"])
s.run(["pip", "install", "-r", "requirements.txt"])
