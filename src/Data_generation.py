""" This code generates a database from a rancge of corrected images  """



import random as rd
import numpy as np
import cv2
#from line_selection.corrected_images.horizontal_lines_detection import detect_lines, filterBetterLines
from detection import select_points, register_points
from extract_image import extract_image_video
from perspective.correction_perspective import *
from bgr_to_rgb import *
from skimage import transform
#import glob

vid0 = "..\\data\\videos\\vid0_clean"
vid1 = "..\\data\\videos\\100NL_FAF.mov_clean"


def rotateImage(I, rot1, zoom, display):
    """
    
    Returns a rotated image

    Args:
        I (array): the input image

        rot1 (integer): the angle of rotation

        zoom (float): the zoom of the output image
        
        display (bool): if we want to display the output image of not
        
    
    """
    h,w = I.shape[:2]
    rotate= cv2.getRotationMatrix2D((w/2, h/2), rot1, zoom)
    rotated_image= cv2.warpAffine(I, rotate, (w, h))
    
    if display:
        cv2.imshow("Rotation of " + str(rot1) + "degrees", rotated_image)
    
    return rotated_image



def crop_image(I, crop_window, display):
    """
    
    crops the image 
    
    Args:
        I (array): the input image
        
        crop_window (list): size of the box of the output image (list of four intergers)
        
        display (bool): if we want to display the output image of not
        
    Output:
        image of size (crop_image[1]-crop_image[0], crop_image[3]-crop_image[2])
        
        
    """
    newI=I[crop_window[0]:crop_window[1],crop_window[2]:crop_window[3]]
    
    if display:
        cv2.imshow(" Cropped image ", newI)
        
    return newI




def Horizontal_flip(I, display):
    """
    
    Gives the flip image 
    
    Args:
        I (array): the input image
        
        display (bool): if we want to display the output image of not
    
    """
    if display:
        cv2.imshow("Horizontal flip", cv2.flip(I,1))
        
    return cv2.flip(I,1)


def translate(I, x, y, display):
    """
    Translates the image, moving the left corner position
    
    Args:
        I (array) : 
        x (integer) : 
        y (integer) : 
        display (bool) : 
    
    """
    h, w = I.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    newI = cv2.warpAffine(I, M, (w, h))
    return newI



def multiplicating_noise(I, alpha, display):
    """
    
    
    """
    h, w = I.shape[:2]
    new_I = I.copy()
    
    for i in range(h):
        for j in range(w):
            new_I[i][j][0] = int(I[i][j][0]*(alpha + (1 - alpha) * rd.random()))
            new_I[i][j][1] = int(I[i][j][1]*(alpha + (1 - alpha) * rd.random()))
            new_I[i][j][2] = int(I[i][j][2]*(alpha + (1 - alpha) * rd.random()))
    return new_I



def add_noise(I, display):
    """
    
    
    """
    h, w = I.shape[:2]
    new_I = I.copy()
    
    for i in range(h):
        for j in range(w):
            new_I[i][j][0] = (I[i][j][0]+rd.randint(-2, 2))%255
            new_I[i][j][1] = (I[i][j][1]+rd.randint(-2, 2))%255
            new_I[i][j][2] = (I[i][j][2]+rd.randint(-2, 2))%255
    return new_I




def shear_image(I, shear, display):
     """
     
     Shears the image
     
     Args:
         I (array)
         shear (float)
         display (bool)
     
     """
     affine = transform.AffineTransform(shear = shear)
     newI = transform.warp(I, affine)
     return newI


def generateImageData(I, index = 0, name_video = ' '):
    """
    
    Generates and save new images that were slightly modified from the input one
    
    Args:
        I (array) : the input warped image we generate image data from
        
        index (integer) : used to name the saved image
        
        name_video (string) : name of the video from which the image was extracted (to name the saved outputd) 
        
    Output:
        None
    
    """
    h, w = I.shape[:2]
    
    #crops the image
    imA = crop_image(I,[int(h/4), int(3*h/5), int(w/4), int(3*w/5)], False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_crop1.jpg", imA)
    imB = crop_image(I,[int(h/4), int(4*h/5), int(w/6), int(3*w/5)], False)
    cv2.imwrite("..\\data\\images\\full_pool\\"+ name_video + str(index) + "_crop2.jpg", imB)
    
    #flips the image
    imC = Horizontal_flip(I, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_flip.jpg", imC)
    
    #translates the image
    imD = translate(I, int(h/8), int(w/8), False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_translate1.jpg", imD)
    imE = translate(I, int(h/7), int(w/4), False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_translate2.jpg", imE)
    
    #image convolution
    imF = multiplicating_noise(I, 0.7, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_mult_noise1.jpg",imF)
    imG = multiplicating_noise(I, 0.78, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_mult_noise2.jpg",imG)
    imH = multiplicating_noise(I, 0.85, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_mult_noise3.jpg",imH)
    
    #adds noise to the image
    imI = add_noise(I, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_add_noise1.jpg",imI)
    imJ = add_noise(imI, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_add_noise2.jpg",imJ)
    
    #shears the image
    imK = shear_image(I, 0.1, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_shear1.jpg",imK)
    imL = shear_image(I, 0.3, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_shear2.jpg",imL)
    imM = shear_image(I, 0.5, False)
    cv2.imwrite("..\\data\\images\\full_pool\\" + name_video + str(index) + "_shear3.jpg",imM)




if __name__ == "__main__":
    
    #We first extract random images from several different videos  
    time_begin0 = 3
    time_end0 = 12
    list_cleanImg0 = extract_image_video(vid0, time_begin0, time_end0, False)
    time_begin1 = 7
    time_end1 = 18
    list_cleanImg1 = extract_image_video(vid1, time_begin1, time_end1, False)
    
    time0 = time_end0 - time_begin0
    time1 = time_end1 - time_begin1
    
    #list of the selected images index
    for k in range(5):
        index_image0 = rd.randint(0, time0 * 24)
        index_image1 = rd.randint(0, time1 * 24)
        I0 = list_cleanImg0[index_image0]
        I1 = list_cleanImg0[index_image1]
        generateImageData(I0, k, name_video = 'vid0')
        generateImageData(I1, k, name_video = 'vid1')
        