import os

import numpy as np

from deconvolution import compute_psnr, gaussian_kernel, wiener_filtering


def test_filtering_constant():
    dirname = os.path.dirname(__file__)
    original_img = np.load(os.path.join(dirname, "original_img.npy"))
    noisy_img = np.load(os.path.join(dirname, "noisy_img.npy"))

    kernel = gaussian_kernel(size=15, sigma=5)
    filtered_img = wiener_filtering(noisy_img, kernel)

    psnr_filtered = compute_psnr(filtered_img, original_img)
    psnr_noisy = compute_psnr(noisy_img, original_img)
    assert psnr_filtered - psnr_noisy > 7.0
