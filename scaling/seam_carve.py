import numpy as np

def compute_energy(img: np.ndarray)->np.ndarray:

    Y = img[:,:,0] *  0.299 + img[:,:,1] *  0.587 + img[:,:,2] *  0.114
    H, W = Y.shape
    X_grad = np.zeros_like(Y)
    Y_grad = np.zeros_like(Y)
    Y = np.pad(Y, ((1, 1), (1, 1)), mode='edge')

    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if i != 1 and i != H:
                X_grad[i - 1, j - 1] = (Y[i + 1, j] - Y[i - 1, j]) / 2
            else:
                X_grad[i - 1,j - 1] = Y[i + 1, j] - Y[i - 1, j]

            if j != 1 and j != W:
                Y_grad[i - 1, j - 1] = (Y[i, j + 1] - Y[i, j - 1]) / 2
            else:
                Y_grad[i - 1, j - 1] = Y[i, j + 1] - Y[i, j - 1]

    energy = np.sqrt(X_grad ** 2 + Y_grad ** 2, dtype='float64')

    return energy


def compute_seam_matrix(energy :np.ndarray, mode: str, mask=None)->np.ndarray:

    H, W = energy.shape
    if mask is not None:
        energy_mask = (np.zeros_like(mask, dtype='float64') + mask) * H * W * 256
        energy += energy_mask
    seam_matrix = np.zeros_like(energy, dtype='float64')

    if mode == 'horizontal':
        seam_matrix[0, :] = energy[0, :]
        for i in range(1, H):
            for j in range(W):
                min_neighboor = None
                if j != 0 and j != W - 1:
                    min_neighboor = np.min([seam_matrix[i - 1, j - 1], seam_matrix[i - 1, j], seam_matrix[i - 1, j + 1]])
                elif j == 0:
                    min_neighboor = np.min([seam_matrix[i - 1, j], seam_matrix[i - 1, j + 1]])
                elif j == W - 1:
                    min_neighboor = np.min([seam_matrix[i - 1, j], seam_matrix[i - 1, j - 1]])
                seam_matrix[i, j] = energy[i,j] + min_neighboor
    elif mode == 'vertical':
        seam_matrix[:, 0] = energy[:, 0]
        for j in range(1, W):
            for i in range(H):
                min_neighboor = None
                if i != 0 and i != H - 1:
                    min_neighboor = np.min([seam_matrix[i - 1, j - 1], seam_matrix[i, j - 1], seam_matrix[i + 1, j - 1]])
                elif i == 0:
                    min_neighboor = np.min([seam_matrix[i, j - 1], seam_matrix[i + 1, j - 1]])
                elif i == H - 1:
                    min_neighboor = np.min([seam_matrix[i - 1, j - 1], seam_matrix[i, j - 1]])
                seam_matrix[i, j] = energy[i, j] + min_neighboor

    return seam_matrix


def remove_minimal_seam(image :np.ndarray, seam_matrix :np.ndarray, mode :str, img_mask = None):
    H, W = seam_matrix.shape

    mask = np.zeros_like(seam_matrix, dtype='uint8')
    remove_seam_img = None
    remove_seam_mask = None

    image = image.astype('uint8')
    if mode == 'horizontal shrink':
        remove_seam_img = np.zeros((H, W - 1, 3), dtype='uint8')
        if img_mask is not None:
            remove_seam_img_mask = np.zeros((H, W - 1), dtype='int8')
        min_idx = np.argmin(seam_matrix[-1])
        for i in range(H - 1, -1, -1):
            diff = None
            if min_idx != 0 and min_idx != W - 1:
                diff = np.argmin(seam_matrix[i, min_idx - 1: min_idx + 2]) - 1
            elif min_idx == W - 1:
                diff = np.argmin(seam_matrix[i, min_idx - 1:]) - 1
            elif min_idx == 0:
                diff = np.argmin(seam_matrix[i, :min_idx + 2]) - 1

            if min_idx == 0 and diff == 0:
                min_idx += 1
            elif min_idx != 0:
                min_idx += diff

            mask[i][min_idx] = 1
            remove_seam_img[i] = np.concatenate((image[i, :min_idx, :], image[i, min_idx + 1:, :]), axis=0, dtype='uint8')
            if img_mask is not None:
                remove_seam_img_mask[i] = np.concatenate((img_mask[i, :min_idx], img_mask[i, min_idx + 1:]), axis=0, dtype='int8')

    elif mode == 'vertical shrink':
        remove_seam_img = np.zeros((H - 1, W, 3), dtype='uint8')
        if img_mask is not None:
            remove_seam_img_mask = np.zeros((H - 1, W), dtype='int8')
        min_idx = np.argmin(seam_matrix[:,-1])
        for j in range(W - 1, -1, -1):
            diff = None
            if min_idx != 0 and min_idx != H - 1:
                diff = np.argmin(seam_matrix[min_idx - 1: min_idx + 2, j]) - 1
            elif min_idx == H - 1:
                diff = np.argmin(seam_matrix[min_idx - 1:, j]) - 1
            elif min_idx == 0:
                diff = np.argmin(seam_matrix[:min_idx + 2, j]) - 1

            if min_idx == 0 and diff == 0:
                min_idx += 1
            elif min_idx != 0:
                min_idx += diff

            mask[min_idx][j] = 1
            remove_seam_img[:, j] = np.concatenate((image[:min_idx, j, :], image[min_idx + 1:, j, :]), axis=0, dtype='uint8')
            if img_mask is not None:
                remove_seam_img_mask[:, j] = np.concatenate((img_mask[:min_idx, j], img_mask[min_idx + 1:, j]), axis=0, dtype='int8')


    return remove_seam_img, remove_seam_mask, mask


def seam_carve(image: np.ndarray, mode: str, mask=None):
    energy = compute_energy(image)
    orientation = mode.split(sep=" ")[0]

    seam_matrix = compute_seam_matrix(energy, orientation, mask)
    remove_seam_img, removed_seam_mask, seam_mask = remove_minimal_seam(image, seam_matrix, mode, mask)

    return remove_seam_img, removed_seam_mask, seam_mask
