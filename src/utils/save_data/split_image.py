import cv2
from src.utils.save_data.crop_lines import crop


def split_and_save(image, margin, destination, frame, nb_lines):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    list_y = [int(1/nb_lines * i * image.shape[0]) for i in range(1, nb_lines)]
    list_images = crop(image, list_y, margin)
    for c in range(1, nb_lines - 1):
        name = 'l%d' % c + '_f' + '0' * (4 - (len(str(frame)))) + str(frame) + '.jpg'
        cv2.imwrite(str(destination / name), list_images[c - 1])
