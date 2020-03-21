"""
To get the image with all the swimmers and all the rectangle boxes
"""


from merge_lines import *
from red_spectrum import *
#from calibration import *


def boxes(name):
    frames, list_y = load_lines(name, "vid0_clean", 0, 0)
    rectangles = []
    for frame, y in zip(frames, list_y):
        c, s = get_rectangle(keep_edges(frame, 2, False), [0, y])
        rectangles.append([c, s])

    image = merge(frames)
    plt.imshow(image)
    for r in rectangles:
        draw_rectangle(r[0], r[1], True)
    plt.imsave("grostest.jpg")
    plt.show()


if __name__ == "__main__":
    boxes("test0.jpg")

