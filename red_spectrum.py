"""
This code allows to indentify a swimmer on an photo with the red spectrum method.
@author: Victoria Brami, Maxime Brisinger, Theo Vincent
"""

import numpy as np 
# this is the key library for manipulating arrays. Use the online ressources! http://www.numpy.org/

import matplotlib.pyplot as plt 
# used to read images, display and plot. 

import scipy.ndimage as ndimage
# one of several python libraries for image procession

plt.rcParams['image.cmap'] = 'gray' 
# by default, the grayscale images are displayed with the jet colormap: use grayscale instead

name = 'man.jpg'
threshold = 0.9
threshold2 = 10

def load_image(name, crop_window=-1): 
    I = plt.imread(name)
    I = I[:,:,0]
    I = I.astype('float')/255 # just to scale the values of the image between 0 and 1 (instead of 0 255)
    return I

def compute_gradient(I, sigma=0):
    I = ndimage.gaussian_filter(I, sigma, mode = 'constant', cval = 0)
    v = np.array([[1./9, 0, -1./9],
               [2./9, 0, -2./9],
               [1./9, 0, -1./9]])
    h = np.array([[1./9, 2./9, 1./9],
               [0, 0, 0],
               [-1./9, -2./9, -1./9]])
    Iv = ndimage.convolve(I, v)
    Ih = ndimage.convolve(I, h)
    In = np.sqrt(Iv**2 + Ih**2)
    return (Iv, Ih, In)

I = load_image(name)
I_threshold = (I > threshold) * 255
I_grady, I_gradx, I_gradnorm = compute_gradient(I_threshold, 3)
#I_gradnorm = I_gradnorm > threshold2
#I_edges = I_gradnorm > threshold
plt.figure(figsize=(5,5)) # this line is not necessary, but allows you to control the size of the displayed image
plt.imshow(I_gradnorm)
plt.show()

def extreme_white_pixels(I):
    I = I > threshold2
    y_min, x_min = I.shape[0] - 1, I.shape[1] - 1
    y_max, x_max = 0, 0
    for y in range(I.shape[0] - 1):
        for x in range(I.shape[1] -1):
            if I[y,x]:
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
    return (x_min, y_min),  (x_max, y_max)

extremes = extreme_white_pixels(I_gradnorm)
size = (extremes[1][0] - extremes[0][0], extremes[1][1] - extremes[0][1])

rectangle = plt.Rectangle(extremes[0], size[0], size[1], fc="none", ec="red")

I = plt.imread(name)
plt.figure(figsize=(5,5)) # this line is not necessary, but allows you to control the size of the displayed image
plt.imshow(I)

plt.gca().add_patch(rectangle)
plt.show()