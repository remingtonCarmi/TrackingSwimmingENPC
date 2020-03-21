"""
To get the image with all the swimmers and all the rectangle boxes
"""


from merge_lines import *
from red_spectrum import *
#from calibration import *
from calibration import make_video


def boxes1(name, margin):
    l_frames, list_y = load_lines(name, "vid0_clean", 0, 0)
    frames = l_frames[0]
    rectangles = []

    for frame, y in zip(frames[1:-1], list_y[1:-1]):
        frame = bgr_to_rgb(frame)[margin:-margin, :]
        c, s = get_rectangle(keep_edges(frame, 2, False), [0, y])
        rectangles.append([c, s])

    image = merge(frames)
    image = bgr_to_rgb(image)
    plt.figure()
    plt.imshow(image)
    for r in rectangles:
        draw_rectangle(r[0], r[1], True)
    plt.show()


def boxes(name, margin, time_begin, time_end):
    lines_per_frame, list_y = load_lines(name, "vid0_clean", time_begin, time_end)
    rectangles = []
    images = []
    count = 0
    size_y = np.shape(lines_per_frame[0][0])[1]
    print(size_y)
    for frame in lines_per_frame:
        rectangles_per_frame = []
        for line, y in zip(frame[1:-1], list_y[1:-1]):
            line = bgr_to_rgb(line)[margin:-margin, :]
            c, s = get_rectangle(keep_edges(line, 2, False), [0, y])
            rectangles_per_frame.append([c, s])
        rectangles.append(rectangles_per_frame)

        image = merge(frame)
        image = bgr_to_rgb(image)
        fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        plt.axis('off')
        plt.imshow(image)
        for r in rectangles_per_frame:
            if abs(r[1][0]) < size_y / 4:
                draw_rectangle(r[0], r[1], True)

        real_name = 'test\\t\\fig%d.jpg' % count

        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(real_name, bbox_inches="tight", transparent=True, pad_inches=0)
        plt.close()

        count += 1

        image_plot = cv2.imread(real_name)
        images.append(image_plot)

    make_video("test\\video_boxes.mp4", images)


if __name__ == "__main__":
    boxes("frame2.jpg", 8, 0, 1.9)

