import cv2

video = cv2.VideoCapture('vid0.mp4')
(success, image) = video.read()
count_image = 0

while success and count_image < 3:
    cv2.imwrite("frame%d.jpg" % count_image, image)
    (success, image) = video.read()
    print('Read a new frame: ', success)
    count_image += 1
