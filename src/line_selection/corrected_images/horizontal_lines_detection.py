# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:07:13 2020

@author: Victoria
"""

import cv2
import numpy as np
from extract_image import extract_image_video
from operator import itemgetter


#Use of unwarped and undistored images
list_cleanImg = extract_image_video("videos\\vid0_clean", 3, 5, False)
cleanImg = list_cleanImg[0]


#two ways to remove the lines that we don't need in the image

def form_clustering(lines : list, threshold : int, average : list)  -> list:
    """
    
    First way to filter some lines we don't want to have. The goal is to remove
    the clusters of lines ans keep a unique one for the group
    
    lines : list containing the two points at the border of the image for each
            line
    threshold : minimum distance (pixels) needed between two lines (if the 
                distance is to low, we delete one of the lines)        
    average : list containing each line's slope value
    tr
    
    """
    cluster = lines.copy()
    new_av = average.copy()
    
    for i in range(1, len(lines)):
        
        #to mesure the closeness with other lines: 0 if it is too close
        proxi = [0] * len(lines)
        proxi[i] = 1
        
        for j in range(len(cluster)):
            [x1,y1] = lines[i][0]
            [x2,y2] = cluster[j][0]
            norm = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if norm >= threshold:
                proxi[j] = 1
        
        for k in range(len(proxi)):
            count_zero = 0
            if proxi[k] == 0:
                count_zero += 1
        
        if count_zero >= 1:
            cluster.remove(lines[i])
            new_av.remove(average[i])
    return cluster, new_av



def form_clustering2(lines : list, threshold : int, average : list) -> list:
    """
    
    Second way to filter some lines we don't want to have. We first decrease
    the number of occurences of  line to one, then we remove the other lines 
    that seem too close.
    
    lines : list containing the two points at the border of the image for each
            line
    threshold : minimum distance needed between two lines (if the distance 
                is to low, we delete one of the lines)        
    average : list containing each line's slope value
    
    
    """
    newlines = lines.copy()
    new_av = average.copy()
    k = 0
    while k < len(newlines):
        a = newlines[k]
        while newlines.count(a) > 1:
            newlines.remove(a)
            new_av.remove(average[k])
        for line in newlines:
            x1, y1 = line[0]
            if line != a:
                if abs(y1-a[0][1]) < threshold:
                    new_av.remove(average[newlines.index(line)])
                    newlines.remove(line)            
        k += 1
    return newlines, new_av




def detect_lines(I2, name_window : str, 
                 threshold : int) -> list:
    """
     
    threshold : parameter for filtering the number of lines
    
    """
    h, w = I2.shape[:2]
    edges = cv2.Canny(I2,50,150,apertureSize = 3)
    NewI = I2.copy()
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180, 400, minLineLength, maxLineGap)
    average = []
    select_lines = []
    Norms = []
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            norm = np.sqrt((x1-x2)**2+(y1-y2)**2)
            Norms.append(norm)
    
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            
            #parameters of the line
            a = (y2 - y1)/(x2 - x1)
            b = y1 - a * x1
            average.append(a)
            select_lines.append([(0,int(b)),(w,int(a*w+b))])
    
    #filtering only the essential lines
    newlines, new_av = form_clustering2(select_lines, threshold, average)
    
    #filtering once more to delete remaining duplicates
    for l in newlines:
        y3 = l[0][1]
        for s in newlines:
            y4 = s[0][1]
            if s != l and abs(y3 - y4) < 10:
                newlines.remove(s)
                
          
    #making sure the line is almost horizontal regardint 
    for i in range(len(newlines)):
        cv2.line(NewI, newlines[i][0], newlines[i][1],(255,100,230),2)
    
    #Show the image    
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)        
    cv2.imshow(name_window, NewI)
    cv2.imwrite("test\\HorizontalLines.jpg", NewI)
    print("nombre de lignes: ", len(newlines))
    
    #sorting the lines
    newlines = sorted(newlines, key = itemgetter(1))
    print(newlines)
    return NewI, newlines




if __name__ == "__main__":
    name_window = "Horizontal lines"
    newI, lines = detect_lines(cleanImg, name_window, 80)