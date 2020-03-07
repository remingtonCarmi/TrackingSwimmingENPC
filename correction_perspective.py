# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:05:04 2020

@author: Victoria
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread("frame25.jpg", 1)
im2 = cv2.imread("frame25.jpg")


def onclick(event):
    print('button=%d, position_x=%f, position_y=%f' %
          (event.button, event.xdata, event.ydata))


fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.imshow(im2)

def correctPerspective(img, src, dst, testing):
    h, w = img.shape[:2]
    # we find the transform matrix M thanks to the matching of the four points
    M = cv2.getPerspectiveTransform(src, dst)
    # warp the image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Corrected Image', fontsize=30)
        plt.show()
    else:
        return warped, M




w, h = im.shape[0], im.shape[1]
w2, h2 = im2.shape[0], im2.shape[1]
print("largeur ", w2)
print("hauteur ", h2)
# We will first manually select the source points 
# we will select the destination point which will map the source points in
# original image to destination points in unwarped image
src1 = np.float32([(20,     1),
                  (540,  130),
                  (20,    520),
                  (570,  450)])

dst1 = np.float32([(1100, 0),
                  (0, 0),
                  (1100, 350),
                  (0, 350)])

#unwarp(im, src1, dst1, True)

#src = np.float32([(478, 714), (1774, 828),
#                  (486, 693), (1821, 821),
#                  (499, 670), (1860, 808),
#                  (507, 643), (1926, 799)])
#    
#dst = np.float32([(0,0), (600,0),
#                  (0, 10), (600, 10),
#                  (0, 20), (600, 20),
#                  (0,30), (600, 30)])

src = np.float32([(44.458678, 313.036059),
                  (976.666271, 231.601832),
                  (81.604114, 410.185663),
                  (1175.965826, 265.889928)
        ])
    
dst = np.float32([ (0,400), (0,0),
                  (20, 400), (20, 0)])

correctPerspective(im2, src, dst1, True)
correctPerspective(im2, src, dst1, False)