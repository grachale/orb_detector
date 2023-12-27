"""
orb algorithm
"""
from typing import List
from typing import Tuple
import cv2
import numpy as np
from scipy.signal import convolve2d
from utils import apply_gaussian_2d


FAST_CIRCLE_RADIUS = 3
FAST_ROW_OFFSETS = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3]
FAST_COL_OFFSETS = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
FAST_FIRST_TEST_INDICES = [0, 4, 8, 12]
FAST_FIRST_TEST_THRESHOLD = 3
FAST_SECOND_TEST_THRESHOLD = 12


def create_pyramid(
    img: np.ndarray, n_pyr_layers: int, downscale_factor: float = 1.2
) -> List[np.ndarray]:
    """
    Creates multi-scale image pyramid.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    n_pyr_layers : int
        Number of layers in the pyramid.
    downscale_factor: float
        Downscaling performed between successive pyramid layers.

    Returns
    -------
    pyr : List[np.ndarray]
        Pyramid of scaled images.
    """
    pyr = [img]

    for i in range(1, n_pyr_layers):
        # calculate the new size for the next layer
        new_size = (
            int(img.shape[1] / (downscale_factor ** i)),
            int(img.shape[0] / (downscale_factor ** i))
        )
        resized_img = cv2.resize(img, new_size)
        pyr.append(resized_img)

    return pyr

def calculate_kp_scores(
    img_level: np.ndarray,
    keypoints: List[Tuple[int, int]],
) -> List[int]:
    """
    Calculates FAST score for initial keypoints.

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    keypoints: List[Tuple[int, int]]
        Tentative keypoints detected by FAST algorithm.

    Returns
    -------
    scores : List[int]
        Scores for the tentative keypoints.
    """
    img_level = img_level.astype(int)
    scores = []
    # go through all keypoints
    for row, column in keypoints:
        # create ring
        ring = []
        for index in zip(FAST_ROW_OFFSETS, FAST_COL_OFFSETS):
            ring.append(img_level[row + index[0]][column + index[1]])

        # minimums of absolute differences
        minimums = []
        for _ in range(0, 16):
            differences = []
            for i in range(0, 9):
                differences.append(np.abs(img_level[row][column] - ring[i]))
            minimums.append(min(differences))
            ring = np.roll(ring, 1)
        scores.append(int(max(minimums)))
    return scores

def test_pixel(center_pixel: int, border_pixel: int, threshold: int):
    """
    Test if the center pixel intensity is significantly different from the border pixel intensity.

    This function compares the intensity of the center pixel with the intensity of the border pixel,
    considering a specified threshold. It returns True if the absolute intensity difference is greater
    than the threshold, indicating a significant difference in intensity; otherwise, it returns False.

    Parameters:
    - center_pixel (int): Intensity of the center pixel.
    - border_pixel (int): Intensity of the border pixel.
    - threshold (int): Threshold for intensity difference.

    Returns:
    - (bool, bool): First element is for dark intensity, the second one if for light.
    """
    border_dark_pixel = max(border_pixel - threshold, 0)
    border_light_pixel = min(border_pixel + threshold, 255)
    dark = False
    light = False
    if center_pixel < border_dark_pixel:
        dark = True
    if center_pixel > border_light_pixel:
        light = True
    return dark, light

def get_first_test_mask(
    img_level: np.ndarray, threshold: int, border: int
) -> np.ndarray:
    """
    Returns the mask from the first FAST test (FAST_FIRST_TEST_INDICES).

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    mask : np.ndarray
        Boolean mask with True values at pixels which pass the first FAST test.
    """
    img_level = img_level.astype(int)

    # Get the shape of the image
    rows, cols = img_level.shape

    # create the mask of the image filled with falses
    mask = np.full((rows, cols), False, dtype=bool)

    for row in range(0 + border, rows - border):
        for column in range(0 + border, cols - border):
            first_test_threshold_dark = 0
            first_test_threshold_light = 0
            for index in FAST_FIRST_TEST_INDICES:
                dark, light = test_pixel(img_level[row][column], img_level[row + FAST_ROW_OFFSETS[index]][column + FAST_COL_OFFSETS[index]], threshold)
                if dark:
                    first_test_threshold_dark += 1
                if light:
                    first_test_threshold_light += 1
            if (first_test_threshold_dark >= FAST_FIRST_TEST_THRESHOLD) or (first_test_threshold_light >= FAST_FIRST_TEST_THRESHOLD):
                mask[row][column] = True
    return mask

def get_second_test_mask(
    img_level: np.ndarray,
    first_test_mask: np.ndarray,
    threshold: int,
    border: int,
) -> np.ndarray:
    """
    Returns the mask from the second FAST test (FAST_FIRST_TEST_INDICES).
    HINT: test only at those points which already passed the first test (first_test_mask).

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    first_test_mask: np.ndarray
        Boolean mask for the first test, which was created by get_first_test_mask().
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    mask : np.ndarray
        Boolean mask with True values at pixels which pass the second FAST test.
    """
    img_level = img_level.astype(int)

    # Get the shape of the image
    rows, cols = img_level.shape

    for row in range(0 + border, rows - border):
        for column in range(0 + border, cols - border):
            if first_test_mask[row][column]:
                second_test_threshold_dark = 0
                second_test_threshold_light = 0
                for index in zip(FAST_ROW_OFFSETS, FAST_COL_OFFSETS):
                    dark, light = test_pixel(img_level[row][column], img_level[row + index[0]][column + index[1]], threshold)
                    if dark:
                        second_test_threshold_dark += 1
                    if light:
                        second_test_threshold_light += 1
                if (second_test_threshold_dark < FAST_SECOND_TEST_THRESHOLD) and (second_test_threshold_light < FAST_SECOND_TEST_THRESHOLD):
                    first_test_mask[row][column] = False
    return first_test_mask

def detect_keypoints(
    img_level: np.ndarray,
    threshold: int,
    border: int = 0,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Creates the initial keypoints list.

    Scans the image at the given pyramid level and detects the unfiltered FAST keypoints,
    which are upscaled according to the current level index.

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    keypoints : List[Tuple[int, int]]
        Initial FAST keypoints as tuples of (row_idx, col_idx).
    scores: List[int]
        Corresponding scores calculate with calculate_kp_scores().
    """
    border = max(border, FAST_CIRCLE_RADIUS)
    keypoints, scores = [], []

    # finding keypoints
    first_test_matrix = get_first_test_mask(img_level, threshold, border)
    second_test_matrix = get_second_test_mask(img_level, first_test_matrix, threshold, border)
    # finding indexes of keypoints
    rows, cols = second_test_matrix.shape
    for row in range(0 + border, rows - border):
        for column in range(0 + border, cols - border):
            if second_test_matrix[row][column]:
                keypoints.append((row, column))
    scores = calculate_kp_scores(img_level, keypoints)
    return keypoints, scores


def get_x_derivative(img: np.ndarray) -> np.ndarray:
    """
    Calculates x-derivative by applying separable Sobel filter.
    HINT: np.pad()

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    result : np.ndarray
        X-derivative of the input image.
    """
    img = img.astype(int)
    # Sobel filter for x-derivative
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])

    result = convolve2d(img, kernel, mode='valid')

    return np.pad(result, 1, mode='constant', constant_values=0)

def get_y_derivative(img: np.ndarray) -> np.ndarray:
    """
    Calculates y-derivative by applying separable Sobel filter.
    HINT: np.pad()

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    result : np.ndarray
        Y-derivative of the input image.
    """
    img = img.astype(int)
    # Sobel filter for y-derivative
    kernel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    result = convolve2d(img, kernel, mode='valid')

    return np.pad(result, 1, mode='constant', constant_values=0)


def get_harris_response(img: np.ndarray) -> np.ndarray:
    """
    Calculates the Harris response.

    Calculates ixx, ixy and iyy from x and y-derivatives with Gaussian
    windowing (utils.apply_gaussian_2d(data=..., sigma=1.0). Then, uses the
    computed matrices to calculate the determinant and trace of the second-
    moment matrix. From it, calculates the final Harris response.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    harris_response : np.ndarray
        Harris response of the input image.
    """
    dx, dy = get_x_derivative(img), get_y_derivative(img)
    dx, dy = dx.astype(float) / 255.0, dy.astype(float) / 255.0
    ixx = dx ** 2
    ixy = dx * dy
    iyy = dy ** 2
    ixx_smoothed = apply_gaussian_2d(1.0, ixx)
    ixy_smoothed = apply_gaussian_2d(1.0, ixy)
    iyy_smoothed = apply_gaussian_2d(1.0, iyy)

    det_a = ixx_smoothed * iyy_smoothed - ixy_smoothed ** 2
    trace_a = ixx_smoothed + iyy_smoothed
    harris_response = det_a - 0.05 * trace_a ** 2
    return harris_response

def filter_keypoints(
    img: np.ndarray, keypoints: List[Tuple[int, int]], n_max_level: int
) -> List[Tuple[int, int]]:
    """
    Filters keypoints by Harris response.

    Iterates the detected keypoints for the given level. Sorts those keypoints
    by their Harris response in the descending order. Then, takes only the
    n_max_level top keypoints.

     Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    keypoints : List[Tuple[int, int]]
        Initial FAST keypoints.
    n_max_level : int
        Maximal number of keypoints for a single pyramid level.

    Returns
    -------
    filtered_keypoints : List[Tuple[int, int]]
        Filtered FAST keypoints.
    """
    harris_response = get_harris_response(img)
    keypoints_values = []
    for row, column in keypoints:
        keypoints_values.append((harris_response[row][column], (row, column)))

    sorted_keypoints_values = sorted(keypoints_values, key=lambda x: x[0], reverse=True)
    filtered_keypoints = []
    for i in range(0, min(len(sorted_keypoints_values), n_max_level)):
        filtered_keypoints.append(sorted_keypoints_values[i][1])

    return filtered_keypoints


def fast(
    img: np.ndarray,
    threshold: int = 20,
    n_pyr_levels: int = 8,
    downscale_factor: float = 1.2,
    n_max_features: int = 500,
    border: int = 0,
) -> List[List[Tuple[int, int]]]:
    """
    Applies the modified FAST detector.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    threshold: int
        Absolute intensity threshold for FAST detector.
    n_pyr_levels : int
        Number of layers in the image pyramid.
    downscale_factor: float
        Downscaling performed between successive pyramid layers.
    n_max_features : int
        Total maximal number of keypoints.
    """
    pyr = create_pyramid(img, n_pyr_levels, downscale_factor)
    keypoints_pyr = []
    # Adapt Nmax for each level
    factor = 1.0 / downscale_factor
    n_max_level, n_sum_levels = [], 0
    n_per_level = n_max_features * (1 - factor) / (1 - factor**n_pyr_levels)
    for level in range(n_pyr_levels):
        n_max_level.append(int(n_per_level))
        n_sum_levels += n_max_level[-1]
        n_per_level *= factor
    n_max_level[-1] = max(n_max_features - n_sum_levels, 0)
    for level, img_level in enumerate(pyr):
        keypoints, scores = detect_keypoints(img_level, threshold, border=border)
        idxs = np.argsort(scores)[::-1]
        keypoints = np.asarray(keypoints)[idxs][: 2 * n_max_level[level]].tolist()
        keypoints = filter_keypoints(img_level, keypoints, n_max_level[level])
        upscale_factor = downscale_factor**level
        keypoints = [
            (int(x * upscale_factor), int(y * upscale_factor)) for (x, y) in keypoints
        ]
        keypoints_pyr.append(keypoints)
    return keypoints_pyr
