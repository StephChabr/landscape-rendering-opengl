# landscape-rendering-opengl

This repository contains most of the files I developped during my internship.

-utils.py: contains functions useful for camera parameters selection

-textureLoader.py and camera.py: contain methods to load our terrain in OpenGL and move freely in it

-select_cameras.py and select_cameras2.py: take camera paraemters and check if thery coorespond to an acceptable view or not

-create_inputs.py: create the input images that will be used to train the network

-explore_world.py: used for debugging

-created_world.py and created_world2.py: use the model obtained from our training and performs the inference, allowing to move in the hallucinated terrain
