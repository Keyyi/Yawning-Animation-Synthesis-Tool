# Yawning Animation Sythesis Tool 

For the course CSC2521, 3D Facial Modeling and Animation, we introduce a yawning animation synthesis tool. 
We provide the user with a UI in Maya where they can specify related parameters, including mouth openness level, yawning duration,
mouth crookedness level, and eye open/closed status, and then generate a yawning animation on ARKit Face Blendshapes.
<p align="center">
<img src="https://github.com/Keyyi/Yawning-Animation-Synthesis-Tool/assets/55814020/cbf9a0c5-fe1d-47e4-ba57-d92ca8152e93" width="280" position='middle' />
<p/>

## Running the Code

**Clone:**
```
git clone https://github.com/Keyyi/Yawning-Animation-Synthesis-Tool.git
```

**Packages:**

```
cd Yawning-Animation-Synthesis-Tool
```
* Replace ```/Applications/Autodesk/maya2022/Maya.app/Contents/bin/mayapy``` with the path of 'mayapy' executable for Maya 2022 on your laptop.
```
/Applications/Autodesk/maya2022/Maya.app/Contents/bin/mayapy -m pip install -r requirements.txt
* Restart Maya to make sure all packages are successfully installed.
```

**Running:**

* Change line 112 and 115 in MayaUI.py to the path of file ```code/model/eye_data.json``` and ```code/model/models.pickle```.
* Open ```code/apple_face.md``` in Maya and set to 60 fps.
* Copy the code from ```code/MayaUI.py``` to Maya python shell and execute!
