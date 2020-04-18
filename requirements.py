"""
This code allows to download useful libraries to launch the project.
"""

# import the os library

import os

# download library for video processing

os.system("conda install python")
os.system("pip install opencv-python")
os.system("pip install PyQt5")

# create needed folders 

"""if not os.path.exists("test"):
    os.makedirs("test")"""

