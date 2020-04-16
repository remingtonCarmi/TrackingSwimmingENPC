
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.extract_image import extract_image_video
from src.detection import select_points
from src.bgr_to_rgb import bgr_to_rgb



def correct_perspective_img(img, src, dst, testing, display):
    """
    
    Affects four selected points in an input image to four other chosen in order 
    to obtain an image with corrected perspective
    
    Args:
        img (array): the input image
        src (list of four tuples): the four points selected on the input image to correct perspective
        dst (list of four tuples): the coordinates of the four points selected in the output image
        testing (bool)
        display (bool): if we want to display the image or not
    Returns
        the perspective-corrected image
        
    """
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


    


if __name__ == "__main__":
    vid0 = "..\\..\\data\\videos\\vid0"
    list_images = extract_image_video(vid0, 4, 5, False)
    cv2.imwrite("..\\..\\test\\imageTest1.jpg", list_images[0])
    im = cv2.imread("..\\..\\test\\imageTest1.jpg")
    points = select_points(im)
    im = bgr_to_rgb(im)
    src = np.float32([(points[0][0], points[0][1]),
                      (points[1][0], points[1][1]),
                      (points[3][0], points[3][1]),
                      (points[2][0], points[2][1])
                      ])
    print(src)
    
    #choosing the right correction points to have non streched images
    d_1 = np.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)
    d_2 = np.sqrt((points[1][0] - points[2][0])**2 + (points[1][1] - points[2][1])**2)
    d = min(d_1, d_2)
    
    dst2 = np.float32([(1500, 0),
                  (0, 0),
                  (1500, d),
                  (0, d)])
    print("Select four points to unwarp the perspective: they must form a quadrangle")
    new_im = correct_perspective_img(im, src, dst2, True, False)

