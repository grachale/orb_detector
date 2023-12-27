import math
import numpy as np


def get_kernel_size(sigma):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    ksize += int(ksize % 2 == 0)
    return int(ksize)


def get_gaussian_kernel(sigma):
    kernel_size = get_kernel_size(sigma)
    # fill the x-values for the kernel
    x_values = np.linspace(
        -math.ceil(3.0 * sigma), math.ceil(3.0 * sigma), kernel_size
    )  # (np.linspace(min_value, max_value, n_elements))
    # fill the corresponding kernel values
    kernel = np.exp(-np.square(x_values) / (2 * sigma * sigma)) * (
        1 / (math.sqrt(2 * math.pi) * sigma)
    )
    return kernel, x_values


def transpose(data):
    if len(data.shape) == 3:
        data = np.transpose(data, (1, 0, 2))
    elif len(data.shape) == 2:
        data = data.T
    else:
        raise ValueError
    return data


def apply_gaussian_1d(sigma, data, horizontal=True):
    kernel, _ = get_gaussian_kernel(sigma)
    if horizontal:  # If a vertical / transposed filter is to be used
        data = transpose(data)
    data_filtered_1d = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 0, data
    )
    if horizontal:
        data_filtered_1d = transpose(data_filtered_1d)

    return data_filtered_1d


def apply_gaussian_2d(sigma, data):
    data_filtered_1d = apply_gaussian_1d(sigma, data, horizontal=True)
    data_filtered_2d = apply_gaussian_1d(sigma, data_filtered_1d, horizontal=False)
    return data_filtered_2d
