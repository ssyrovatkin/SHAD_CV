import numpy as np

def edge_crops(img):
    h, w = img.shape
    delta_h = int(0.1 * h)
    delta_w = int(0.1 * w)
    img = img[delta_h:-delta_h, delta_w:-delta_w]

    return img

def find_shift(shift, img_size):
    if abs(img_size - shift) <= shift and (shift - img_size <= 0) :
        return shift - img_size
    else:
        return shift

def align(img, g_coord):

    h, w = img.shape
    h = h-int(h % 3)
    img = img[:h, :]

    r_channel = edge_crops(img[:int(h / 3),:])
    g_channel = edge_crops(img[int(h / 3) : int(2 * h / 3), :])
    b_channel = edge_crops(img[int(2 * h / 3):, :])

    r_dft_matrix = np.fft.fft2(r_channel)
    b_dft_matrix = np.fft.fft2(b_channel)
    g_dft_matrix = np.fft.fft2(g_channel)

    C_gr = np.fft.ifft2(r_dft_matrix * np.conjugate(g_dft_matrix))
    C_gb = np.fft.ifft2(b_dft_matrix * np.conjugate(g_dft_matrix))

    gr_shift_idx = np.unravel_index(np.argmax(C_gr, axis=None), C_gr.shape)
    gb_shift_idx = np.unravel_index(np.argmax(C_gb, axis=None), C_gb.shape)

    r_row = find_shift(gr_shift_idx[0], r_channel.shape[0]) + g_coord[0] - int(h / 3)
    r_col = find_shift(gr_shift_idx[1], r_channel.shape[1]) + g_coord[1]

    b_row = find_shift(gb_shift_idx[0], b_channel.shape[0]) + g_coord[0] + int(h / 3)
    b_col = find_shift(gb_shift_idx[1], b_channel.shape[1]) + g_coord[1]

    g_coord_new = int(h / 3) + int(0.1 * h), int(0.1 * w)

    # r_borders = img[find_shift(gb_shift_idx[0], b_channel.shape[0]) + g_coord_new[0] + int(h / 3):
    #                 find_shift(gb_shift_idx[0], b_channel.shape[0]) + g_coord_new[0] + g_channel.shape[0] + int(h / 3),
    #                 find_shift(gb_shift_idx[1], b_channel.shape[1]) + g_coord_new[1]:
    #                 find_shift(gb_shift_idx[1], b_channel.shape[1]) + g_coord_new[1] + g_channel.shape[1]]
    #
    # b_borders = img[find_shift(gr_shift_idx[0], r_channel.shape[0]) + g_coord_new[0] - int(h / 3):
    #                 find_shift(gr_shift_idx[0], r_channel.shape[0]) + g_coord_new[0] + g_channel.shape[0] - int(h / 3),
    #                 find_shift(gr_shift_idx[1], r_channel.shape[1]) + g_coord_new[1]:
    #                 find_shift(gr_shift_idx[1], r_channel.shape[1]) + g_coord_new[1] + g_channel.shape[1]]
    # #
    # result_img = np.dstack([r_borders, g_channel, b_borders])

    result_img = np.dstack([np.roll(b_channel, (-find_shift(gb_shift_idx[0], b_channel.shape[0]),
                                                -find_shift(gb_shift_idx[1], b_channel.shape[1]))),
                            g_channel, np.roll(r_channel, (-find_shift(gr_shift_idx[0], r_channel.shape[0]),
                                                -find_shift(gr_shift_idx[1], r_channel.shape[1])))])


    return result_img, (r_row, r_col), (b_row, b_col)