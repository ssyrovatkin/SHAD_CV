import numpy as np
from scipy.signal import convolve2d


def get_bayer_masks(n_rows: int, n_cols: int):
    r_mask, g_mask, b_mask = (np.zeros((n_rows, n_cols)), np.zeros((n_rows, n_cols)), np.zeros((n_rows, n_cols)))
    for i in range(n_rows):
        for j in range(n_cols):
            if i % 2 == 0:
                if j % 2 == 1:
                    r_mask[i][j] = 1
                elif j % 2 == 0:
                    g_mask[i][j] = 1
            elif i % 2 == 1:
                if j % 2 == 0:
                    b_mask[i][j] = 1
                elif j % 2 == 1:
                    g_mask[i][j] = 1
    mask = np.dstack((r_mask, g_mask, b_mask)).astype(bool)

    return mask


def get_colored_img(raw_img):
    img = np.array(raw_img, dtype='uint8')
    n_rows, n_cols = img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    colored_img = np.zeros((n_rows, n_cols, 3), dtype='uint8')
    for i in range(3):
        colored_img[:,:,i] = np.array(masks[:,:,i] * img, dtype='uint8')

    return colored_img


def bilinear_interpolation(colored_img):

    h, w, c = colored_img.shape
    interpolated_img = np.zeros_like(colored_img, dtype='uint8')
    masks = get_bayer_masks(h, w)
    inverse_mask = np.logical_not(masks)

    for i in range(1, h):
        for j in range(1, w):
            pixels = colored_img[i - 1:i + 2, j - 1:j + 2, :]
            mask = masks[i - 1:i + 2, j - 1:j + 2, :]
            value = np.sum(pixels, axis=(0,1)) / np.sum(mask, axis=(0,1))
            interpolated_img[i][j] = value

    interpolated_img *= inverse_mask
    interpolated_img += colored_img

    return interpolated_img


def improved_interpolation(raw_img):
    h, w = raw_img.shape
    RGB_masks = get_bayer_masks(h, w)
    R_m, G_m, B_m = RGB_masks[:,:,0], RGB_masks[:,:,1], RGB_masks[:,:,2]

    GR_GB = np.array([[0.0, 0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, 2.0, 0.0, 0.0],
                      [-1.0, 2.0, 4.0, 2.0, -1.0],
                      [0.0, 0.0, 2.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0, 0.0]], dtype='float64') / 8.0

    Rg_RB_Bg_BR = np.array([[0.0, 0.0, 0.5, 0.0, 0.0],
                            [0.0, -1.0, 0.0, -1.0, 0.0],
                            [-1.0, 4.0, 5.0, 4.0, -1.0],
                            [0.0, -1.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.5, 0.0, 0.0]], dtype='float64') / 8.0

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = np.array([[0.0, 0.0, -1.5, 0.0, 0.0],
                            [0.0, 2.0, 0.0, 2.0, 0.0],
                            [-1.5, 0.0, 6.0, 0.0, -1.5],
                            [0.0, 2.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, -1.5, 0.0, 0.0]], dtype='float64') / 8.0

    R, G, B = raw_img * R_m, raw_img * G_m, raw_img * B_m

    del G_m

    G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve2d(raw_img, GR_GB, mode='same', boundary='fill', fillvalue=0), G)

    RBg_RBBR = convolve2d(raw_img, Rg_RB_Bg_BR, mode='same', boundary='fill', fillvalue=0)
    RBg_BRRB = convolve2d(raw_img, Rg_BR_Bg_RB, mode='same', boundary='fill', fillvalue=0)
    RBgr_BBRR = convolve2d(raw_img, Rb_BB_Br_RR, mode='same', boundary='fill', fillvalue=0)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    R_c = np.any(R_m == 1, axis=0)[None] * np.ones(R.shape)

    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
    B_c = np.any(B_m == 1, axis=0)[None] * np.ones(B.shape)

    del R_m, B_m

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    return np.clip(np.dstack([R, G, B]), 0, 255).astype(np.uint8)

def  compute_psnr(img_pred, img_gt):
    img_pred = np.array(img_pred, dtype='float64')
    img_gt = np.array(img_gt, dtype='float64')
    mse = np.mean((img_pred - img_gt)**2)
    if mse != 0:
        psnr = 10 * np.log10(np.max(img_gt**2) / mse)
    else:
        raise(ValueError)
    return psnr
