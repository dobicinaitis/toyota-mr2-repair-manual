# Helper functions

import os
import re
import shlex
import sys

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path


def expand_path(path):
    """
    Expand the ~ character to the user's home directory.
    :param path: a path starting with ~/
    :return: an expanded path
    """
    path_parts = shlex.split(path)
    unescaped_path = ' '.join(path_parts)
    expanded_path = os.path.expanduser(unescaped_path)
    return expanded_path


def is_pdf_file(path):
    """
    Check if the path points to a PDF file.
    :param path: file path
    :return: true if the file is a PDF file, false otherwise
    """
    if not os.path.isfile(path):
        return False
    _, ext = os.path.splitext(path)
    return ext.lower() == '.pdf'


def parse_arguments():
    """
    Parse command line arguments and environment variables.
    :return: a tuple containing the manual path, page number, and image output directory
    """
    page_number = 0
    image_output_directory = '.'  # default > current directory
    manual_path = os.environ.get('MR2_DOCS_MANUAL_PATH')
    usage_message = '\nUsage: python extract_illustrations.py <page-number> [output-directory] [path-to-pdf]'
    argument_count = len(sys.argv)

    if argument_count >= 4:
        manual_path = sys.argv[3]
    elif manual_path is None:
        print('Error: Path to the manual PDF file is required.')
        print(usage_message)
        sys.exit(1)

    manual_path = expand_path(manual_path)
    if not is_pdf_file(manual_path):
        print(f"Error: Path '{manual_path}' is not a PDF file.")
        print(usage_message)
        sys.exit(1)

    if argument_count >= 3:
        image_output_directory = sys.argv[2]
        if not os.path.exists(image_output_directory):
            print(f'Error: The provided output directory {image_output_directory} does not exist.')
            print(usage_message)
            sys.exit(1)

    if argument_count >= 2:
        page_number = int(sys.argv[1])
        if page_number < 1:
            print('Error: The page number must be a positive integer.')
            print(usage_message)
            sys.exit(1)
    else:
        print(usage_message)
        sys.exit(1)

    print("Using arguments:")
    print(f"Manual path: {manual_path}")
    print(f"Page number: {page_number}")
    print(f"Image output directory: {image_output_directory}")

    return manual_path, page_number, image_output_directory


def pdf_to_image(pdf_path, page_number):
    """
    Convert the specified PDF page to an image (NumPy array).
    :param pdf_path: path to the PDF file
    :param page_number: page number to convert
    :return: NumPy array representing the image
    """
    image = convert_from_path(pdf_path, dpi=600, first_page=page_number, last_page=page_number)[0]
    return np.array(image)


def show_image(image, window_name='ImageWindow'):
    """
    Display an image in a resizable window.
    :param image: image to show
    :param window_name: optional window name
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ImageWindow', 600, 800)
    cv2.imshow(window_name, image)

    while True:
        # break the loop when the window is closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        # break the loop when "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)


def order_points(points):
    """
    Order a rectangle edge coordinates as follows: top-left, top-right, bottom-right, and bottom-left.
    :param points: coordinates of the rectangle edges
    :return: ordered coordinates
    """
    rectangle = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rectangle[0] = points[np.argmin(s)]
    rectangle[2] = points[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]

    return rectangle


def compute_distance(a, b):
    """
    Compute the Euclidean distance between two points.
    :param a: coordinates of the first point
    :param b: coordinates of the second point
    :return: distance between the points
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def level_rectangle(image, points):
    """
    Apply a perspective transformation to level an image.
    :param image: image to transform
    :param points: rectangle edge coordinates
    :return: a transformed image
    """
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    width_a = compute_distance(bottom_right, bottom_left)
    width_b = compute_distance(top_right, top_left)
    max_width = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = compute_distance(top_right, bottom_right)
    height_b = compute_distance(top_left, bottom_left)
    max_height = max(int(height_a), int(height_b))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def get_illustrations(image):
    """
    Detect and extract illustrations (bordered rectangles) from an image.
    :param image: an image containing one or multiple illustrations
    :return: a list of extracted illustrations
    """
    detected_illustrations = []

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 1)
    image_canny = cv2.Canny(image_blur, 50, 50)

    # find contours in the image
    contours, _ = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100_000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            corner_count = len(approx)

            if corner_count == 4:
                # draw the contours on the image for debugging
                cv2.drawContours(image, contour, -1, (0, 255, 0), 3)

                edge_points = np.empty((0, 2), int)
                for point in approx:
                    x, y = point[0]
                    edge_points = np.append(edge_points, [[x, y]], axis=0)

                illustration = level_rectangle(image, edge_points)
                detected_illustrations.append(illustration)
    print(f"Detected {len(detected_illustrations)} illustrations.")
    return detected_illustrations


def get_filename(image):
    """
    Compute the filename from illustration IDs.
    The image ID is located in the bottom right corner of the illustration
    and consists of one or more strings in format: XX0000
    :param image: image containing the illustration
    :return: "_" separated string of IDs, e.g. "AB1234_CD5678"
    """
    ids = []
    image_height, image_width, _ = image.shape
    # approximate area where the title is located
    id_box_height = 80
    id_box_width = 630
    if image_width > id_box_width:
        id_box_height *= 2

    box_top_left_coordinates = (image_width - id_box_width, image_height - id_box_height)
    box_bottom_right_coordinates = (image_width, image_height)

    # image_with_boxed_ids = image.copy()
    # cv2.rectangle(image_with_boxed_ids, box_top_left_coordinates, box_bottom_right_coordinates, (0, 255, 0), 2)
    # show_image(image_with_boxed_ids, "ID box")

    id_box_crop = image[box_top_left_coordinates[1]:box_bottom_right_coordinates[1],
                  box_top_left_coordinates[0]:box_bottom_right_coordinates[0]]

    text = pytesseract.image_to_string(id_box_crop).strip()
    # print(f"Detected text: {text}")

    # remove an extra O letter that sometimes gets aded before 3 zeros during OCR
    text = re.sub(r"([A-Z]{2})(O000)(\d)", r"\g<1>000\g<3>", text)
    # print(f"Odd '0ooo' fix: {text}")

    id_pattern_approximate = r'[A-Z]{2}\w{4}'  # two uppercase letters followed by four alphanumeric characters
    id_pattern_specific = r'[A-Z]{2}\d{4}'  # two uppercase letters followed by four digits

    misidentified_digit_map = {
        'O': '0',
        'I': '1',
        'S': '5',
        'Z': '2',
        'B': '8',
        'G': '6',
        'Q': '0',
        'D': '0'
    }

    maybe_matches = re.findall(id_pattern_approximate, text)
    for match in maybe_matches:
        fuzzy_part = match[2:].upper()

        # try to replace letters with possible mis-identified digits
        for letter, digit in misidentified_digit_map.items():
            fuzzy_part = fuzzy_part.replace(letter, digit)

        match = match[:2] + fuzzy_part

        # check if the match is now valid
        if re.match(id_pattern_specific, match):
            # print(f"Matched title pattern: {match}")
            ids.append(match)

    # helper.show_image(title_box_subsection)
    filename = "_".join(ids)
    print(f"Illustration filename: {filename}")
    return filename


def remove_border(image, border_thickness):
    """
    Remove the border from an image.
    :param image: image with a border
    :param border_thickness: thickness of the border
    :return: an image without the border
    """
    return image[border_thickness:-border_thickness, border_thickness:-border_thickness]


def white_to_alpha(image):
    """
    Convert the white pixels in an image to transparent ones.
    :param image: image to process
    :return: an image with white pixels converted to transparent ones
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    # create a mask where white is 0 (transparent) and black is 255 (opaque)
    mask = np.where(binary_image == 255, 0, 255).astype(np.uint8)
    # create a new 4-channel image (BGRA)
    result = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGRA)
    # set the alpha channel to the mask
    result[:, :, 3] = mask
    return result
