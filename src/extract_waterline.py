# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:21:15 2020

@author: Victoria
"""

import numpy as np
from extract_image import extract_image_video
from horizontal_lines_detection import detect_lines
from calibration import make_video


if __name__ == "__main__":
    
    list_cleanImg = extract_image_video("videos\\vid0_clean", 3, 15, False)
    cleanImg = list_cleanImg[0]
    h, w = cleanImg.shape[:2]
    name_window = "Horizontal lines"
    newI, lines = detect_lines(cleanImg, name_window, 80)
    new_list_img = []
    print("Quelle ligne d'eau voulez-vous ? ")
    line_nb = int(input())
    assert(line_nb in list(range(1,10)))
    print(lines[line_nb-1][1])
    src = np.float32([lines[line_nb-1][0], lines[line_nb-1][1],
                    lines[line_nb][0], lines[line_nb][1]])
    dst = np.float32([(w, 100),(0, 100), (w, 300), (0, 300)])
    
    for i in range(len(list_cleanImg)):
        I_line = list_cleanImg[i][lines[line_nb-1][1][1] : lines[line_nb][1][1]+1, : ]
        new_list_img.append(I_line)
    name_video_clean = "videos\\vid0_cleanLine" + str(line_nb) + ".mp4"
    make_video(name_video_clean,  new_list_img)
    #cv2.imshow("One waterline", I_line)