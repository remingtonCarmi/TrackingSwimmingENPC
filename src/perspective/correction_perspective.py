# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:05:04 2020

@author: Victoria
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from extract_image import extract_image_video
from distortion import find_distortion_charact, clear_image, SelectionError
from extract_image import TimeError
from detection import select_points, register_points

vid0 = "videos\\vid0"

def convertRGBtoBGR(I):
    newI = I.copy()
    newI[:, :, 0], newI[:, :, 2] = I[:, :, 2], I[:, :, 0]
    return newI

def correctPerspectiveImg(img, src, dst, testing, display):
    h, w = img.shape[:2]
    # we find the transform matrix M thanks to the matching of the four points
    M = cv2.getPerspectiveTransform(src, dst)
    # warp the image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        if display:
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
            return cv2.flip(warped, 1)
        else:
            return cv2.flip(warped, 1)
    else:
        return warped, M
    

# We will first manually select the source points 
# we will select the destination point which will map the source points in
# original image to destination points in unwarped image
src1 = np.float32([(20,     1),
                  (540,  130),
                  (20,    520),
                  (570,  450)])

    
dst2 = np.float32([(1500, 0),
                  (0, 0),
                  (1500, 750),
                  (0, 750)])



    
if __name__ == "__main__":
    list_images = extract_image_video(vid0, 0, 5, False)
    cv2.imwrite("imageTest1.jpg", list_images[0])
    im = cv2.imread("imageTest1.jpg")
    points = select_points(im)
    im = convertRGBtoBGR(im)
    src = np.float32([(points[0][0], points[0][1]),
                      (points[1][0], points[1][1]),
                      (points[3][0], points[3][1]),
                      (points[2][0], points[2][1])
                      ])
    print(src)
    new_im = correctPerspectiveImg(im, src, dst2, True, True)