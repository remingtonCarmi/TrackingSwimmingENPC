"""
This code allows to download useful libraries to launch the project.
@author: Victoria Brami, Maxime Brisinger, Theo Vincent
"""

# import the os library

import os

# download library for video processing

os.system("pip install opencv-python")

# create needed folders 

if not os.path.exists("test"):
    os.makedirs("test")

if not os.path.exists("videos"):
    os.makedirs("videos")