import numpy as np
from scipy.fft import fft2, fftn


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = (1/ np.sqrt(2*np.pi*np.square(sigma))) * np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h_shape = h.shape
    ph, pw = shape[0] - h_shape[0], shape[1] - h_shape[1]

    padding = [((ph + 1) // 2, ph // 2), ((pw + 1) // 2, pw // 2)]
    pad_h = np.pad(h, padding, mode='constant', constant_values=0)
    pad_h = np.fft.fftshift(pad_h)

    if shape[0] % 2 == 1:
      pad_h = np.roll(np.roll(pad_h, 1, axis=0), 1, axis=1)
    fft_h = np.fft.fft2(pad_h)

    return fft_h


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    mask = np.absolute(H) <= threshold
    H_inv = (1 / (mask + H)) * (1 - mask)
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    pass
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    F = G * inverse_kernel(H, threshold)
    f = np.fft.ifftshift(np.absolute(np.fft.ifft2(F)))
    if f.shape[0] % 2 == 1:
        f = np.roll(np.roll(f, -1, axis=0), -1, axis=1)

    return f


def wiener_filtering(blurred_img, h, K=6e-5):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    F = np.conjugate(H) / (np.absolute(H) ** 2 + K) * G
    f = np.fft.ifftshift(np.absolute(np.fft.ifft2(F)))
    if f.shape[0] % 2 == 1:
        f = np.roll(np.roll(f, -1, axis=0), -1, axis=1)

    return f


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr
