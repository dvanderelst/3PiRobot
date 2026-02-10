# Goals of this code

This folder contains code for controlling a Pololu 3pi+ 2040 Robot equipped with 3 maxbotix MB1360 XL-MaxSonar-AEL0 sonar sensors. One acts as an emitter, and the other two act as a receiver (ears). The receivers provide the analog envelope of the received echoes. The robot has an RP2040 microcontroller programmed in Python.

MB1360 Datasheet: https://maxbotix.com/products/mb1360

# Robot code

The folder Robot_code/upload contains code that runs on the robot. 

The folder Robot_code/upload/lib/pololu_3pi_2040_robot contains the Pololu library for working with the robot. Other files in the lib folder (Robot_code/upload/lib/) were developed by me to make working with the robot easier. The main.py file is run at boot time and calls the default_algorithm.py code.

Rules for working with the code in the Robot_code folder:

+ Because this is code that is supposed to run on a robot, you can not directly run the code.
+ You can write tests of code using the local virtual environment interpreter (in .venv). However, all test code you create should go in a separate, dedicated test folder next (not inside!) the "upload folder."

# Control code 

The folder Control_code contains code that runs on the computer controlling the robot over a wifi connection. This code specifies a client that controls the robot's movement and acquisition. (/Control_code/Library/Client.py)

Rules for working with the code in the Control_code folder:

+ Since this code controls a robot, you should not run any test directly, as this might move the robot. You can propose tests for me to run.

# General Rules

+ Do not automatically commit edits to git. I will handle all commits manually.


