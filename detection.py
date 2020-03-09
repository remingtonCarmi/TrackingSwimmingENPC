"""The aim of this code is to make the user select the points
that are relevant to withdraw distortion and perspective."""
import cv2


def register_points(event, x_coord, y_coord, flags, param):
    """
    Select the point that have been clicked.

    Args:
        event (event): the action done by the user

        x_coord (float): the x coordinate of the clicked point

        y_coord (float): the y coordinate of the clicked point

        param (list of all the parameters):
            param[0] : list_points (list of lists of floats): the list of selected points

            param[1] : image (array): the image that we study

            param[2] : name_window (string): the name of the window
    """
    # if the left mouse button was clicked, we register the point
    # and we show the selected point
    if event == cv2.EVENT_LBUTTONDOWN:
        # registration of the point selected
        param[0].append([x_coord, y_coord])
        # drawing
        for point in param[0]:
            cv2.circle(param[1], (point[0], point[1]), 2, (0, 0, 255), 2)
        cv2.imshow(param[2], param[1])


def select_points(image):
    """
    Makes the user select the points he wants.
    Click left to select the point,
    press r to withdraw the last point,
    press q to exit.

    Args:
        image (array): the array representing the image

    Returns:
        points (list of lists of floats): the selected points
    """
    # --- Parameters --- #
    points = []
    nb_select_points = 0
    selecting = True

    # load the image, clone it, and setup the mouse callback function
    clone = image.copy()
    name_window = "image_select"
    param = [points, image, name_window]

    while selecting:
        # display the image
        cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
        cv2.imshow(param[2], param[1])
        cv2.setMouseCallback(param[2], register_points, param)
        key = cv2.waitKey(0) & 0xFF

        # we count the number of points that we have selected
        nb_select_points = len(param[0])

        # if the 'r' key is pressed, withdraw the last point selected if it exists
        if nb_select_points > 0 and key == ord("r"):
            # withdraw the last point
            param[0] = param[0][:-1]
            # display the right image
            param[1][:, :] = clone[:, :]
            for point in param[0]:
                cv2.circle(param[1], (point[0], point[1]), 2, (0, 0, 255), 2)
            cv2.imshow(param[2], param[1])

        # if the 'q' key is pressed, exit the loop
        elif key == ord("q"):
            selecting = False

    cv2.destroyAllWindows()

    return param[0]


if __name__ == "__main__":
    print("Click left to select the point, press r to withdraw the last point, press q to exit.")
    IMAGE = cv2.imread("test\\test_img.jpg")
    print(select_points(IMAGE))
