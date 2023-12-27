import math
import cv2
import pytest
import numpy as np
# from scipy.signal import convolve2d
# from scipy.spatial.distance import cdist
from pathlib import Path
# from utils import apply_gaussian_2d
import orb
import inspect
from pylint.lint import Run
from pylint.reporters import CollectingReporter

REF_PATH = Path(__file__).parent / "reference_out"


@pytest.fixture(scope="session")
def linter():
    """Test codestyle for src file of orb detector."""
    src_file = inspect.getfile(orb)
    rep = CollectingReporter()
    # disabled warnings:
    # 0301 line too long
    # 0103 variables name (does not like shorter than 2 chars)
    # E1101 Module 'cv2' has no 'resize' member (no-member)
    # R0913 Too many arguments (6/5) (too-many-arguments)
    # R0914 Too many local variables (18/15) (too-many-locals)
    r = Run(
        ["--disable=C0301,C0103,E1101,R0913,R0914", "-sn", src_file],
        reporter=rep,
        exit=False,
    )
    return r.linter


@pytest.mark.parametrize("limit", range(0, 11))
def test_codestyle_score(linter, limit, runs=[]):
    """Evaluate codestyle for different thresholds."""
    if len(runs) == 0:
        print("\nLinter output:")
        for m in linter.reporter.messages:
            print(f"{m.msg_id} ({m.symbol}) line {m.line}: {m.msg}")
    runs.append(limit)
    # score = linter.stats['global_note']
    score = linter.stats.global_note

    print(f"pylint score = {score} limit = {limit}")
    assert score >= limit


@pytest.fixture(
    name="input_image",
    params=[p for p in (Path(__file__).parent / 'test_images').iterdir() if p.suffix != '.sign'],
    scope="session"
)
def _input_image(request):
    img = cv2.imread(str(request.param))
    return request.param.stem, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@pytest.mark.parametrize(
    "n_pyr_layers,downscale_factor", [[1, 1.0], [2, 4.0], [3, 2.0], [5, 1.2]]
)
def test_create_pyramid(input_image, n_pyr_layers, downscale_factor):
    _, img0 = input_image
    pyr = orb.create_pyramid(img0, n_pyr_layers, downscale_factor)
    assert isinstance(pyr, list)
    assert all(isinstance(img_level, np.ndarray) for img_level in pyr)
    assert len(pyr) == n_pyr_layers
    assert np.array_equal(pyr[0], img0)
    img = img0.copy()
    for i in range(1, n_pyr_layers):
        out_shape = tuple(math.ceil(d / float(downscale_factor)) for d in img.shape)
        pixels_total = out_shape[0] * out_shape[1]
        assert abs(out_shape[0] - pyr[i].shape[0]) <= 2
        assert abs(out_shape[1] - pyr[i].shape[1]) <= 2
        img = cv2.resize(img, pyr[i].shape[::-1])
        pixels_ok = np.isclose(
            img.astype(np.uint8),
            pyr[i].astype(np.uint8),
            atol=5
        ).sum()
        if (pixels_ok / pixels_total) < 0.9:
            img = cv2.resize(img0, pyr[i].shape[::-1])
            pixels_ok = np.isclose(
                img.astype(np.uint8),
                pyr[i].astype(np.uint8),
                atol=5
            ).sum()
        assert (pixels_ok / pixels_total) >= 0.9


# Let student to solve detect_keypoints without helper mask arrays.
# @pytest.mark.parametrize(
#     "threshold,border", [[5, 0], [5, 10], [10, 0], [10, 20], [20, 0], [20, 20]]
# )
# def test_get_first_test_mask(input_image, threshold, border):
#     img_base, img = input_image
#     border = max(border, orb.FAST_CIRCLE_RADIUS)
#     mask = orb.get_first_test_mask(img.astype(int), threshold, border)
#     assert isinstance(mask, np.ndarray)
#     assert mask.shape == img.shape
#     assert mask[:border, :].sum() == 0
#     assert mask[:, :border].sum() == 0
#     assert mask[-border:, :].sum() == 0
#     assert mask[:, -border:].sum() == 0
#     mask_ref = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_get_first_test_mask.npz")['mask_ref']
#     x, y = img.shape
#     assert (
#         np.equal(mask, mask_ref).sum() / x / y > 0.98
#     )  # might depend on dtype whether it is passed as int or np.uint8


# Let student to solve detect_keypoints without helper mask arrays.
# @pytest.mark.parametrize(
#     "threshold,border", [[5, 0], [5, 10], [10, 0], [10, 20], [20, 0], [20, 20]]
# )
# def test_get_second_test_mask(input_image, threshold, border):
#     img_base, img = input_image
#     border = max(border, orb.FAST_CIRCLE_RADIUS)
#     mask1 = orb.get_first_test_mask(img.astype(int), threshold, border)
#     mask = orb.get_second_test_mask(img.astype(int), mask1, threshold)
#     assert isinstance(mask, np.ndarray)
#     assert mask.shape == img.shape
#     assert mask[:border, :].sum() == 0
#     assert mask[:, :border].sum() == 0
#     assert mask[-border:, :].sum() == 0
#     assert mask[:, -border:].sum() == 0
#     mask_ref = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_get_second_test_mask.npz")['mask_ref']
#     x, y = img.shape
#     assert (
#         np.equal(mask, mask_ref).sum() / x / y > 0.98
#     )  # might depend on dtype whether it is passed as int or np.uint8


@pytest.mark.parametrize(
    "threshold,border", [[20, 20]]
)
def test_calculate_kp_scores(input_image, threshold, border):
    img_base, img = input_image
    img = img.astype(int)
    border = max(border, orb.FAST_CIRCLE_RADIUS)
    first_test_mask = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_get_first_test_mask.npz")['mask_ref']
    second_test_mask = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_get_second_test_mask_indices.npz")['mask_ref']
    first_test_passed_r, first_test_passed_c = np.where(first_test_mask)
    keypoints = list(
        zip(
            first_test_passed_r[second_test_mask], first_test_passed_c[second_test_mask]
        )
    )
    scores = orb.calculate_kp_scores(img, keypoints)
    assert isinstance(scores, list)
    assert all([isinstance(score, int) for score in scores])
    scores_ref = (
        np.load(REF_PATH / f"{img_base}_{threshold}_{border}_calculate_kp_scores.npz")['scores_ref']
        .squeeze()
        .tolist()
    )
    assert scores == scores_ref


@pytest.mark.parametrize(
    "threshold,border", [[20, 0], [20, 20]]
)
def test_detect_keypoints(input_image, threshold, border):
    img_base, img = input_image
    border = max(border, orb.FAST_CIRCLE_RADIUS)
    keypoints, scores = orb.detect_keypoints(img, threshold, border)
    keypoints_ref = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_detect_keypoints_1.npz")['keypoints_ref']
    scores_ref = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_detect_keypoints_2.npz")['scores_ref']
    assert isinstance(keypoints, list)
    assert len(keypoints) == len(keypoints_ref)
    assert len(keypoints) == len(scores)
    keypoints, keypoints_ref = np.asarray(keypoints), np.asarray(keypoints_ref)
    scores, scores_ref = np.asarray(scores), np.asarray(scores_ref)
    ind = np.lexsort((keypoints[:, 0], keypoints[:, 1]))
    keypoints, scores = keypoints[ind, :], scores[ind]
    ind_ref = np.lexsort((keypoints_ref[:, 0], keypoints_ref[:, 1]))
    keypoints_ref, scores_ref = keypoints_ref[ind_ref, :], scores_ref[ind]
    assert np.array_equal(keypoints, keypoints_ref)
    assert np.array_equal(scores, scores_ref)


def test_get_x_derivative(input_image):
    img_base, img = input_image
    # compare as float16 for save space of reference array
    result = orb.get_x_derivative(img).astype(np.float16)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    result_ref = np.load(REF_PATH / f"{img_base}_get_x_derivative.npz")['result_ref']
    assert np.allclose(result, result_ref)


def test_get_y_derivative(input_image):
    img_base, img = input_image
    # compare as float16 for save space of reference array
    result = orb.get_y_derivative(img).astype(np.float16)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    result_ref = np.load(REF_PATH / f"{img_base}_get_y_derivative.npz")['result_ref']
    assert np.allclose(result, result_ref)


def test_get_harris_response(input_image):
    img_base, img = input_image
    # compare as float16 for save space of reference array
    response = orb.get_harris_response(img).astype(np.float16)
    assert isinstance(response, np.ndarray)
    assert response.shape == img.shape
    response_ref = np.load(REF_PATH / f"{img_base}_get_harris_response.npz")['response_ref']
    # np.savez_compressed(REF_PATH / f"{img_base}_get_harris_response.npz", response_ref=response_ref)
    assert np.allclose(response, response_ref)


@pytest.mark.parametrize(
    "n_max,threshold,border",
    [
        [50, 20, 0],
        [100, 15, 10],
    ],
)
def test_filter_keypoints(input_image, n_max, threshold, border):
    img_base, img = input_image
    border = max(border, orb.FAST_CIRCLE_RADIUS)
    keypoints_ref = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_detect_keypoints_1.npz")['keypoints_ref']
    scores_ref = np.load(REF_PATH / f"{img_base}_{threshold}_{border}_detect_keypoints_2.npz")['scores_ref']

    idxs = np.argsort(scores_ref)[::-1]
    keypoints_ref = np.asarray(keypoints_ref)[idxs][: 2 * n_max].tolist()
    filtered_keypoints = orb.filter_keypoints(img, keypoints_ref, n_max)
    assert len(filtered_keypoints) <= n_max
    filtered_keypoints = np.asarray(filtered_keypoints)
    filtered_keypoints_ref = np.load(REF_PATH / f"{img_base}_{n_max}_{threshold}_{border}_filter_keypoints.npz")['filtered_keypoints_ref']
    ind = np.lexsort((filtered_keypoints[:, 0], filtered_keypoints[:, 1]))
    ind_ref = np.lexsort((filtered_keypoints_ref[:, 0], filtered_keypoints_ref[:, 1]))
    filtered_keypoints, filtered_keypoints_ref = (
        filtered_keypoints[ind, :],
        filtered_keypoints_ref[ind_ref],
    )
    assert np.array_equal(filtered_keypoints, filtered_keypoints_ref)
