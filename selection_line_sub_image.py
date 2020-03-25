""""
This code allow you to get the lines of the swimmers.
"""

import cv2
import numpy as np
import numpy.linalg as alg
import numpy.random as rd
from matplotlib import pyplot as plt


def canny_edges(image):
    """
    Computes the canny edges of the image.

    Args:
        image (array): the original image.

    Returns:
        image_edges (array, same shape as image_name): the canny edges image.
    """
    return cv2.Canny(image, 100, 200)


def crop_image(image, pourcent_x, pourcent_length, pourcent_y, pourcent_thickness):
    """
    Crops an image.

    Args:
        image (array): the image to crop.

        pourcent_x (float): the returned image will start at pourcent_x * height in the original image
            on the first axis

        pourcent_length (float): the returned image height will be pourcent_length * height

        pourcent_y (float): the returned image will start at pourcent_y * width in the original image
            on the second axis

        pourcent_thickness (float): the returned image width will be pourcent_width * width

    Returns:
        cropped_image (array, smaller shape as image): the cropped image.
    """
    (height, width) = image.shape[: 2]
    x_start = int(pourcent_x * height)
    y_start = int(pourcent_y * width)
    return image[x_start: x_start + int(pourcent_length * height), y_start: y_start + int(pourcent_y * width)]


def in_image(x_coord, y_coord, h_dim, w_dim):
    """
    Says if the point (x_coord, y_coord) is in the image or not.

    Args:
        x_coord (int): the coordinate of the point.

        y_coord (int): the coordinate of the point.

        h_dim (int): the height dimension of the image.

        w_dim (int): the width dimension of the image.

    Returns:
        in_image (boolean):
            True if the point is in the image.

            False if not.
    """
    if 0 <= x_coord < h_dim and 0 <= y_coord < w_dim:
        return True
    return False


def interesting_pixel(x_coord, y_coord, edges, visited):
    """
    Says if the pixel that we consider is interesting.
    That is to say : the pixel is in the image, it is part of
    a canny edge and it has not been visited.

    Args:
        x_coord (int): the x coordinate of the point.

        y_coord (int): the y coordinate of the point.

        edges (array, 2 dimensions, same size as the image):
            edges[i, j] = 1 if (i, j) is part of a canny edge.

        visited (array, 2 dimensions, same size as the image):
            visited[i, j] = 1 if (i, j) has been visited, = 0 if not.

    Returns:
        interesting_pixel (boolean):
            True if the pixel is interesting.

            False if not.
    """
    (height, width) = visited.shape[0: 2]
    if not in_image(x_coord, y_coord, height, width):
        return False
    if not edges[x_coord, y_coord]:
        return False
    if visited[x_coord, y_coord]:
        return False
    return True


def search_neighbors(starting_point, image_edges, visited):
    """
    Search all the points that are interesting near the starting point.

    Args:
        starting_point (couple of int): the coordinate of the strating point.

        image_edges (array, 2 dimensions, same size as the image):
            edges[i, j] = 1 if (i, j) is part of a canny edge.

        visited (array, 2 dimensions, same size as the image):
            visited[i, j] = 1 if (i, j) has been visited, = 0 if not.

    Returns:
        new_line (list of list of int): the list of points in the new line.
    """
    new_line = [starting_point]
    # new_members is a file FIFO
    new_members = [starting_point]
    nb_members = 1
    # Continue until the queue is empty
    while nb_members > 0:
        (index_h, index_w) = new_members.pop(0)
        nb_members -= 1
        for near_h in range(-1, 2):
            for near_w in range(-1, 2):
                h_search = index_h + near_h
                w_search = index_w + near_w
                # If the neighbor is interesting, we save it and
                # we put it in the queue to examine his neighbors
                if interesting_pixel(h_search, w_search, image_edges, visited):
                    visited[h_search, w_search] = 1
                    new_line.append([h_search, w_search])
                    new_members.append([h_search, w_search])
                    nb_members += 1
    return new_line


def extract_lines(image_edges):
    """
    Extracts the lines from the canny edges.

    Args:
        image_edges (array, 2 dimensions, same size as the image):
            edges[i, j] = 1 if (i, j) is part of a canny edge.

    Returns:
        lines (list of lists of lists): the list of the lines.
    """
    list_lines = []
    (height, width) = image_edges.shape
    visited = np.zeros((height, width))

    for index_h in range(height):
        for index_w in range(width):
            # If the pixel has never been visited and is a canny edge we visit it.
            if not visited[index_h, index_w] and image_edges[index_h, index_w]:
                visited[index_h, index_w] = 1
                new_line = search_neighbors([index_h, index_w], image_edges, visited)
                if len(new_line) > SIZE_LINE:
                    list_lines.append(new_line)
    return list_lines


def show_lines(lines, height, width, channel):
    """
    Construct an image to show the lines.

    Args:
        lines (list of lists of lists): the list of the lines.

        height (int): the height of the original image.

        width (int): the width of the original image.

        channel (int): the number of channel of the original image.

    Returns:
        image_lines (array, of shape (height, width, channel)): an black image with the lines.
    """
    image_lines = np.zeros((height, width, channel), np.uint8)
    for line in lines:
        color = rd.randint(50, 255, 3, np.uint8)
        for (coord_x, coord_y) in line:
            image_lines[coord_x, coord_y] = color

    return image_lines


def update_lines(lines, info_reg, interesting_lines):
    """
    Keep the interesting lines and erase the others.

    Args:
        lines (list of lists of lists): the list of the lines.

        info_reg (list of lists of 3 elements): (slope, intercept, accuracy) of each line.

        interesting_lines (array of size = len(lines)):
            interesting_lines[i] = 1 if the line i is interesting, = 0 if not.

    Returns:
        new_lines (list of lists of lists): the list of the updated lines.

        new_info_reg (list of lists of 3 elements): (slope, intercept, accuracy) of each updated line.

        new_interesting_lines (array of size = len(lines)):
            interesting_lines[i] = 1 if the line i is interesting, = 0 if not.
    """
    nb_lines = len(interesting_lines)
    nb_interesting_lines = int(sum(interesting_lines))
    new_lines = [[0, 0]] * nb_interesting_lines
    new_info_reg = [[0, 0, 0]] * nb_interesting_lines
    index_interesting = 0
    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            new_lines[index_interesting] = lines[index_line]
            new_info_reg[index_interesting] = info_reg[index_line]
            index_interesting += 1

    return new_lines, new_info_reg, np.ones(nb_interesting_lines)


def compute_ends(lines, interesting_lines):
    """
    Computes the extremum point of each line according to the width axis.

    Args:
        lines (list of lists of lists): the list of the lines.

        interesting_lines (array of size = len(lines)):
            interesting_lines[i] = 1 if the line i is interesting, = 0 if not.

    Returns:
        ends (array, shape = (len(nb_line), 2, 2)): the list of the extremum points of each line
    """
    nb_lines = len(lines)
    # Array of the left and right point of each line
    ends = np.zeros((nb_lines, 2, 2))
    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            # The extreme left point of the line
            index_left = np.argmin([lines[index_line][i][1] for i in range(len(lines[index_line]))])
            # The extreme point of the line
            index_right = np.argmax([lines[index_line][i][1] for i in range(len(lines[index_line]))])

            ends[index_line, 0] = lines[index_line][index_left]
            ends[index_line, 1] = lines[index_line][index_right]
        else:
            ends[index_line] = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
    return ends


def compute_ends_line(end_line, line_index, line):
    """
    Computes the extremum points of the line according to the width axis.

    Args:
        end_line (array, shape = (len(nb_line), 2, 2)): the list of the extremum points of each line

        line_index (int): the index of the line of which we have to compute the edges

        line (list of lists): the list of the points of line.
    """
    # The extreme left point of the line
    index_left = np.argmin([line[i][1] for i in range(len(line))])
    # The extreme point of the line
    index_right = np.argmax([line[i][1] for i in range(len(line))])

    end_line[line_index, 0] = line[int(index_left)]
    end_line[line_index, 1] = line[int(index_right)]


def join_lines(lines, interesting_lines, end_lines):
    """
    Joins the lines that are closed to each other.

    Notes : recursive function, saved_line is used to avoid deleting element in a list.

    Args:
        lines (list of lists of lists): the list of the lines.

        interesting_lines (array of size = len(lines)):
            saved_lines[i] = 1 if the line i is interesting, = 0 if not.

        end_lines (array, shape = (len(nb_line), 2, 2)): the list of the extremum points of each line
    """
    nb_lines = len(lines)
    # For each line that is intersting, we search if it is worth to join a line.
    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            # Comparison of the point at the left to all the points at the right
            distance = [alg.norm(end_lines[index_line, 0] - end_lines[index_right, 1]) for index_right in range(nb_lines)]
            distance[index_line] = float('inf')
            # If two lines are closed, we add the second line in the first and turn the second one useless
            if min(distance) < JOIN_DISTANCE:
                index_min_dist = np.argmin(distance)
                lines[index_line].extend(lines[index_min_dist])
                interesting_lines[index_min_dist] = 0
                # Upload of edges:
                compute_ends_line(end_lines, index_line, lines[index_line])
                end_lines[index_min_dist] = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
                return join_lines(lines, interesting_lines, end_lines)

            # Comparison of the point at the right to all the points at the left
            distance = np.array([alg.norm(end_lines[index_line, 1] - end_lines[index_left, 0]) for index_left in range(nb_lines)])
            distance[index_line] = float('inf')
            # If two lines are closed, we add the second line in the first and turn the second one useless
            if min(distance) < JOIN_DISTANCE:
                index_min_dist = np.argmin(distance)
                lines[index_line].extend(lines[index_min_dist])
                interesting_lines[index_min_dist] = 0
                # Upload of edges:
                compute_ends_line(end_lines, index_line, lines[index_line])
                end_lines[index_min_dist] = [[float('inf'), float('inf')], [float('inf'), float('inf')]]
                return join_lines(lines, interesting_lines, end_lines)


def linear_regression(x_set, y_set):
    """
    Computes the linear regression of the input set.

    Arg:
        x_set (list of float or int): the list of x_coordinate.

        y_set (list of float or int): the list of y_coordinate.

    Returns:
        coeff (float) : the slope of the regression

        intercept (float): the Y-intercept of the regression.

        accuracy (float): the accuracy of the regression.
    """
    nb_points = len(x_set)
    set_x = np.ones((2, nb_points))
    set_x[0, :] = x_set[:]
    covariance = np.mean((x_set - np.mean(x_set)) * (y_set - np.mean(y_set)))
    accuracy = covariance / (np.std(x_set) * np.std(y_set) + 10 ** -16)
    # (coeff, intercept) = Y * X_t * inv(X * X_t)
    (coeff, intercept) = y_set.dot(np.transpose(set_x).dot(alg.inv(set_x.dot(np.transpose(set_x)))))

    return coeff, intercept, accuracy ** 2


def compute_reg(lines, interesting_lines):
    """
    Computes the linear regression of the lines.

    Args:
        lines (list of lists of lists): the list of the lines.

        interesting_lines (array of size = len(lines)):
            saved_lines[i] = 1 if the line i is interesting, = 0 if not.

    Returns:
        reg_info (list of list of float):
            the list of [slope, intercept, accuracy] of each line.
    """
    nb_lines = len(lines)
    reg_info = [[0, 0, 0]] * nb_lines
    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            line = np.array(lines[index_line])
            reg_info[index_line] = linear_regression(line[:, 1], line[:, 0])

    return reg_info


def compute_reg_line(line):
    """
    Computes the linear regression of the line.

    Args:
        line (list of lists): the list of the points of line.

    Returns:
        coeff (float) : the slope of the regression

        intercept (float): the Y-intercept of the regression.

        accuracy (float): the accuracy of the regression.
    """
    line_array = np.array(line)
    (coeff, intercept, accuracy) = linear_regression(line_array[:, 1], line_array[:, 0])

    return [coeff, intercept, accuracy]


def purify_reg(lines, info_reg, interesting_lines):
    """
    Purifies the lines : get ride of the lines that have a slope too
    far from the average.

    Args:
        lines (list of lists of lists): the list of the lines.

        info_reg (list of lists of 3 elements): (slope, intercept, accuracy) of each line.

        interesting_lines (array of size = len(lines)):
            saved_lines[i] = 1 if the line i is interesting, = 0 if not.
    """
    nb_lines = len(lines)
    coeff_mean = sum([info_reg[index][0] for index in range(nb_lines)]) / nb_lines
    coeff_square_mean = sum([info_reg[index][0] ** 2 for index in range(nb_lines)]) / nb_lines
    coeff_std = np.sqrt(coeff_square_mean - coeff_mean ** 2)

    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            slope = info_reg[index_line][0]
            # If the slope of the line is too far from the mean, we get ride of the line
            if abs(slope - coeff_mean) > coeff_std * NB_STD:
                interesting_lines[index_line] = 0
                coeff_mean -= slope / nb_lines


def show_reg(lines, image, info_reg, impose_slope=True, with_image=True):
    """
    Show the linear regression of the lines.
    If the accuracy of a line is two low, it imposes the mean slope
    and show the linear regression of the line with the imposed slope.

    Args:
        lines (list of lists of lists): the list of the lines.

        image (array): the original image.

        info_reg (list of lists of 3 elements): (slope, intercept, accuracy) of each line.

        impose_slope (optional, boolean):
            if impose_slope = True: all the linear regressions are shown without imposing the slope.

        with_image (optional, boolean):
            if in_image = True: the image and the linear regressions are shown in the same window.

    Returns:
        image_lines (array, shape = (height, width, channel)) a black image with colored lines
            which are the linear regressions.
    """
    nb_lines = len(lines)
    (h_dim, w_dim, nb_channel) = image.shape
    if with_image:
        image_lines = image.copy()
    else:
        image_lines = np.zeros((h_dim, w_dim, nb_channel), np.uint8)
    general_coeff = 0
    sum_accuracy = 0
    # Register the lines that have a good accuracy
    good_lines = np.zeros(nb_lines)

    # If impose_slope = False : all the lines are printed without regarding the accuracy
    level_accuracy = LEVEL_ACCURACY
    if not impose_slope:
        level_accuracy = 0

    for index_line in range(nb_lines):
        (coeff, intercept, accuracy) = info_reg[index_line]
        if accuracy >= level_accuracy:
            good_lines[index_line] = 1
            general_coeff += coeff * accuracy
            sum_accuracy += accuracy
            color = rd.randint(0, 25, 3, np.uint8)
            for y in range(w_dim):
                if in_image(int(y * coeff + intercept), y, h_dim, w_dim):
                    image_lines[int(y * coeff + intercept), y] = color
    mean_coeff = general_coeff / sum_accuracy

    for index_line in range(nb_lines):
        if not good_lines[index_line]:
            line = np.array(lines[index_line])
            # We impose the slope and compute the intercept of the linear regression
            intercept = np.mean(line[:, 0]) - mean_coeff * np.mean(line[:, 1])
            color = rd.randint(0, 25, 3, np.uint8)
            for y in range(w_dim):
                if in_image(int(y * mean_coeff + intercept), y, h_dim, w_dim):
                    image_lines[int(y * mean_coeff + intercept), y] = color
    return image_lines


def integral_difference(info_reg, index_ref, interesting_lines, width):
    nb_lines = len(info_reg)
    nb_interesting_lines = int(sum(interesting_lines))
    differences = np.ones(nb_interesting_lines)
    (slope_ref, intercept_ref) = info_reg[index_ref][: 2]
    nb_interesting = 0

    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            if index_line == index_ref:
                differences[nb_interesting] = float('inf')
            else:
                (slope, intercept) = info_reg[index_line][: 2]
                differences[nb_interesting] = abs(width * (width * (slope - slope_ref) / 2 + intercept - intercept_ref))
            nb_interesting += 1

    return differences


def join_intercept(lines, info_reg, interesting_lines, width):
    """
    Join the lines that are closed to each other.
    For doing so, we compare the integral of the difference on [0, width]

    Notes : recursive function, saved_line is used to avoid deleting element in a list.

    Args:
        lines (list of lists of lists): the list of the lines.

        info_reg (list of lists of 3 elements): (slope, intercept, accuracy) of each line.

        interesting_lines (array of size = len(lines)):
            saved_lines[i] = 1 if the line i is interesting, = 0 if not.

        width (int): the width of the original image.
    """
    nb_lines = len(lines)

    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            # The distance at the left (x = 0) between all the lines
            distance = integral_difference(info_reg, index_line, interesting_lines, width)
            if min(distance) < DTSTANCE_INTEGRAL:
                index_min_dist = np.argmin(distance)
                lines[index_line].extend(lines[index_min_dist])
                interesting_lines[index_min_dist] = 0
                info_reg[index_line] = compute_reg_line(lines[index_line])
                return join_intercept(lines, info_reg, interesting_lines, width)


def select_yellow(lines, image, interesting_lines):
    """
    Select the yellow lines.

    Args:
        lines (list of lists of lists): the list of the lines.

        image (array): the original image.

        interesting_lines (array of size = len(lines)):
            saved_lines[i] = 1 if the line i is interesting, = 0 if not.
    """
    nb_lines = len(lines)
    mean_color = np.array([np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])])

    for index_line in range(nb_lines):
        if interesting_lines[index_line]:
            nb_points = len(lines[index_line])
            sum_r = np.zeros(nb_points)
            sum_g = np.zeros(nb_points)
            sum_b = np.zeros(nb_points)
            for index_points in range(nb_points):
                (x, y) = lines[index_line][index_points]
                (r, g, b) = image[x, y]
                sum_r[index_points] = r
                sum_g[index_points] = g
                sum_b[index_points] = b
            diff_mean = (np.array([np.mean(sum_r), np.mean(sum_g), np.mean(sum_b)] - mean_color))
            # If the line is not yellow
            if diff_mean[0] < 20 or sum(abs(diff_mean)) < 50:
                interesting_lines[index_line] = 0


def select_line(image):
    """
    Select the yellow line from the picture.

    Args:
        image (array): the original image.

    Return:
        show_y (array, the same shape as image): the image with the yellow lines.
    """
    (h_dim, w_dim, nb_channel) = image.shape

    # Computes the canny edges.
    edges_image = canny_edges(image)

    # Extracts the lines.
    list_lines = extract_lines(edges_image)

    # saved_lines registers the interesting lines.
    saved_lines = np.ones(len(list_lines))

    # Joins the lines that are closed according to their ends.
    lines_ends = compute_ends(list_lines, saved_lines)
    join_lines(list_lines, saved_lines, lines_ends)

    # Computes linear regressions of every line.
    reg_info = compute_reg(list_lines, saved_lines)

    # Withdraws the lines that have a slope too far for the average slope.
    purify_reg(list_lines, reg_info, saved_lines)

    # Joins the lines that are closed according to their linear regressions.
    join_intercept(list_lines, reg_info, saved_lines, w_dim)

    # Selects the yellow line.
    select_yellow(list_lines, image, saved_lines)

    (list_lines, reg_info, saved_lines) = update_lines(list_lines, reg_info, saved_lines)
    show_y = show_reg(list_lines, image, reg_info)

    # Displays the image with the lines
    plt.imshow(show_y)
    plt.show()

    return show_y


if __name__ == "__main__":
    IMAGE_PATH = "test.jpg"
    POUCENT_X = 0.3
    POURCENT_LENGTH = 0.4
    POURCENT_Y = 0.3
    POURCENT_THICKNESS = 0.2
    CROP_IMAGE = crop_image(plt.imread(IMAGE_PATH), POUCENT_X, POURCENT_LENGTH, POURCENT_Y, POURCENT_THICKNESS)
    (HEIGHT, WIDTH, CHANNEL) = CROP_IMAGE.shape
    # ---- Parameters ----
    JOIN_DISTANCE = 5  # The number of pixels until which two lines will be join.

    SIZE_LINE = (HEIGHT // 2 + WIDTH // 2) // 3  # The minimum number of pixel in a line.

    LEVEL_ACCURACY = 0.5  # The minimum level of accuracy that we impose to a line to print it.

    # If mean_coeff (resp. mean_std) is the mean (resp. standard deviation) of the regression slopes.
    # We call slope_line the slope of a particular line.
    # This particular line is erased if : abs(slope_line - mean_coeff) > coeff_sdt * RATIO_SLOPE.
    NB_STD = 1

    # The maximum distance for two to join them considering the integral of the difference.
    DTSTANCE_INTEGRAL = WIDTH * 4

    # ---- Selection of the water lines ----
    select_line(CROP_IMAGE)
