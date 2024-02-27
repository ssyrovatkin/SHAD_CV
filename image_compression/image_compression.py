import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    
    # Отцентруем каждую строчку матрицы
    mean_matrix = np.mean(matrix, axis=1)[:,None]
    matrix = matrix - mean_matrix
    # Найдем матрицу ковариации
    covariance_matrix = np.cov(matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Посчитаем количество найденных собственных векторов
    n = len(eigenvalues)
    # Сортируем собственные значения в порядке убывания
    eigenvalues_idx = eigenvalues.argsort()
    sorted_eigenvalues = eigenvalues[eigenvalues_idx[::-1]]
    sorted_eigenvectors = eigenvectors[:, eigenvalues_idx[::-1]]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    sorted_eigenvectors = sorted_eigenvectors[:, :p]
    # Оставляем только p собственных векторов
    compressed_matrix = sorted_eigenvectors.T @ matrix
    # Проекция данных на новое пространство
    
    return sorted_eigenvectors, compressed_matrix, mean_matrix.flatten()


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        decompressed_ch = np.floor(comp[0] @ comp[1] + comp[2][:, None])
        result_img.append(decompressed_ch)

    result_img = np.clip(np.transpose(np.array(result_img), (1,2,0)), 0, 255)
        
    return result_img


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')

    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed_img = pca_compression(img[:,:,j], p)
            compressed.append(compressed_img)
        decompressed = pca_decompression(compressed).astype('uint8')
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    
    # Your code here
    img = img.astype('float64')
    Y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    Cb = -0.1687 * img[:,:,0] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,2] + 128
    Cr = 0.5 * img[:,:,0] - 0.4187 * img[:,:,1] - 0.0813 * img[:,:,2] + 128
    ycbcr_img = np.concatenate((Y[:,:,None], Cb[:,:,None], Cr[:,:,None]), axis=2)

    return ycbcr_img


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    img = img.astype('float64')
    R = img[:,:,0] + 1.402 * (img[:,:,2] - 128)
    G = img[:,:,0] - 0.34414 * (img[:,:,1] - 128) - 0.71414 * (img[:,:,2] - 128)
    B = img[:,:,0] + 1.772 * (img[:,:,1] - 128)
    rgb_img = np.clip(np.concatenate((R[:,:,None], G[:,:,None], B[:,:,None]), axis=2), 0, 255)
    
    return rgb_img.astype('uint8')


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[:, :, :3]

    # Your code here
    ycbcr_img = rgb2ycbcr(rgb_img)
    Y = ycbcr_img[:, :, 0]
    Cb = ycbcr_img[:, :, 1]
    Cr = ycbcr_img[:, :, 2]
    Cb = gaussian_filter(Cb, sigma=10)
    Cr = gaussian_filter(Cr, sigma=10)
    ycbcr_img = np.concatenate((Y[:, :, None], Cb[:, :, None], Cr[:, :, None]), axis=2)
    rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(rgb_img)

    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[:, :, :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    Y = ycbcr_img[:, :, 0]
    Cb = ycbcr_img[:, :, 1]
    Cr = ycbcr_img[:, :, 2]
    Y = gaussian_filter(Y, sigma=10)

    ycbcr_img = np.concatenate((Y[:, :, None], Cb[:, :, None], Cr[:, :, None]), axis=2)
    rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(rgb_img)

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    
    # Your code here
    A, B = component.shape
    component = gaussian_filter(component, sigma=10)
    A_idx = np.arange(0, A, 2)
    B_idx = np.arange(0, B, 2)
    downsampled_component = component[A_idx, :][:,B_idx]
    
    return downsampled_component


def alpha(u):
    if u == 0:
        return 1 / 2**0.5
    else:
        return 1


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    G = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            alpha_u = alpha(u)
            alpha_v = alpha(v)
            sum = 0
            for x in range(8):
                for y in range(8):
                    sum += block[x][y]*np.cos((2*x + 1)*u*np.pi/16)*np.cos((2*y + 1)*v*np.pi/16)
            G[u][v] = alpha_u * alpha_v * sum / 4

    return G

# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    # Your code here
    q_block = np.round(block / quantization_matrix)
    return q_block


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here
    S = None
    if q >= 1 and q < 50:
        S = 5000/q
    elif q == 100:
        S = 1
    else:
        S = 200 - 2*q

    own_q_matrix = np.floor((50 + S * default_quantization_matrix) / 100)
    mask = own_q_matrix == 0
    own_q_matrix += np.ones_like(mask) * mask

    return own_q_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    
    # Your code here
    rows, cols = block.shape
    nums = rows * cols
    zigzag_list = []
    i = 0
    j = 0
    k = 0
    while i < rows and j < cols and k < nums:
        zigzag_list.append(block[i][j])
        k += 1
        if (i+j) % 2 == 0:
            if (i-1) in range(rows) and (j+1) not in range(cols):
                i = i + 1
            elif (i-1) not in range(rows) and (j+1) in range(cols):
                j = j + 1
            elif (i-1) not in range(rows) and (j+1) not in range(cols):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        elif (i+j) % 2 == 1:
            if (i+1) in range(rows) and (j-1) not in range(cols):
                i = i + 1
            elif (i+1) not in range(rows) and (j-1) in range(cols):
                j = j + 1
            elif (i+1) not in range(rows) and (j-1) not in range(cols):
                j = j + 1
            else:
                i = i + 1
                j = j - 1


    zigzag_list = np.array(zigzag_list)
    return zigzag_list


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here
    compressed_zigzag_list = []
    k = 0
    for i, value in enumerate(zigzag_list):
        if value != 0:
            if k != 0:
                compressed_zigzag_list.append(0)
                compressed_zigzag_list.append(k)
                k = 0
            compressed_zigzag_list.append(value)
        elif value == 0:
            k += 1

    if k != 0:
        compressed_zigzag_list.append(0)
        compressed_zigzag_list.append(k)
    compressed_zigzag_list = np.array(compressed_zigzag_list)

    return compressed_zigzag_list


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    
    # Переходим из RGB в YCbCr
    ycbcr_img = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    Y = ycbcr_img[:, :, 0]
    downCb = downsampling(ycbcr_img[:, :, 1])
    downCr = downsampling(ycbcr_img[:, :, 2])
    downsampled_ycbcr_img = np.concatenate((downCb[:,:,None], downCr[:,:,None]), axis=2)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    downsampled_ycbcr_img -= 128
    # Y -= 128
    H, W, _ = downsampled_ycbcr_img.shape

    blocks = [[], [], []]
    for c in range(2):
        for i in range(0, H, 8):
            for j in range(0, W, 8):
                current_block = downsampled_ycbcr_img[i:i+8, j:j+8, c]
                blocks[c + 1].append(current_block)

    H_Y, W_Y = Y.shape

    for i in range(0, H_Y, 8):
        for j in range(0, W_Y, 8):
            current_block = Y[i:i + 8, j:j + 8]
            blocks[0].append(current_block)

    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    updated_blocks = [[], [], []]
    for c in range(3):
        q_matr = None
        if c == 0:
            q_matr = 0
        elif c != 0:
            q_matr = 1
        for block in blocks[c]:
            block = dct(block)
            block = quantization(block, quantization_matrixes[q_matr])
            block = zigzag(block)
            block = compression(block)
            updated_blocks[c].append(block)

    del blocks

    return updated_blocks


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    
    # Your code here
    decompressed_list = []
    k = 0
    for i in range(len(compressed_list)):
        value = compressed_list[i]
        if value != 0:
            if k == 1:
                k = 0
                continue
            elif k == 0:
                decompressed_list.append(value)
        elif value == 0:
            num_zeros = compressed_list[i+1]
            decompressed_list += [0]*int(num_zeros)
            k = 1

    decompressed_list = np.array(decompressed_list)

    return decompressed_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    # Your code here
    block = np.zeros((8, 8))
    rows, cols = 8, 8
    nums = rows * cols

    i = 0
    j = 0
    k = 0
    while i < rows and j < cols and k < nums:
        block[i][j] = input[k]
        k += 1
        if (i + j) % 2 == 0:
            if (i - 1) in range(rows) and (j + 1) not in range(cols):
                i = i + 1
            elif (i - 1) not in range(rows) and (j + 1) in range(cols):
                j = j + 1
            elif (i - 1) not in range(rows) and (j + 1) not in range(cols):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        elif (i + j) % 2 == 1:
            if (i + 1) in range(rows) and (j - 1) not in range(cols):
                i = i + 1
            elif (i + 1) not in range(rows) and (j - 1) in range(cols):
                j = j + 1
            elif (i + 1) not in range(rows) and (j - 1) not in range(cols):
                j = j + 1
            else:
                i = i + 1
                j = j - 1

    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    
    # Your code here
    
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    G = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            sum = 0
            for u in range(8):
                for v in range(8):
                    alpha_u = alpha(u)
                    alpha_v = alpha(v)
                    sum += alpha_u*alpha_v*block[u][v]*np.cos((2*x + 1)*u*np.pi/16)*np.cos((2*y + 1)*v*np.pi/16)
            G[x][y] = sum / 4

    G = np.round(G)

    return G


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    # Your code here
    A, B = component.shape
    new_A = int(2*A)
    new_B = int(2*B)
    upsampled_component = np.zeros((new_A, new_B))
    for i in range(new_A):
        for j in range(new_B):
            upsampled_component[i,j] = component[i // 2, j // 2]

    return upsampled_component


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here
    H, W, _ = result_shape
    H_in = H // 2
    W_in = W // 2

    intermid_result = np.zeros((H, W, 3))
    intermid_color_result = np.zeros((H_in, W_in, 2))

    inverse_blocks = [[], [], []]

    for c in range(3):
        q_matr = None
        if c == 0:
            q_matr = 0
        elif c != 0:
            q_matr = 1
        for block in result[c]:
            block = inverse_compression(block)
            block = inverse_zigzag(block)
            block = inverse_quantization(block, quantization_matrixes[q_matr])
            block = inverse_dct(block)
            inverse_blocks[c].append(block)

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            idx_in_row = int((i // 8) * (W // 8)) + int(j // 8)
            intermid_result[i:i + 8, j:j + 8, 0] = inverse_blocks[0][idx_in_row]

    for c in range(2):
        for i in range(0, H_in, 8):
            for j in range(0, W_in, 8):
                idx_in_row = int((i // 8) * (W_in // 8)) + int(j // 8)
                intermid_color_result[i:i+8, j:j+8, c] = inverse_blocks[c + 1][idx_in_row] + 128

    intermid_result[:, :, 1] = upsampling(intermid_color_result[:, :, 0])
    intermid_result[:, :, 2] = upsampling(intermid_color_result[:, :, 1])
    result_rgb = ycbcr2rgb(intermid_result)

    return result_rgb


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[:, :, :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        result_shape = img.shape
        own_color_quantization_matrix = own_quantization_matrix(color_quantization_matrix, p)
        own_y_quantization_matrix = own_quantization_matrix(y_quantization_matrix, p)
        quantization_matrixes = [own_y_quantization_matrix, own_color_quantization_matrix]
        compressed_img = jpeg_compression(img, quantization_matrixes)
        decompressed_img = jpeg_decompression(compressed_img, result_shape, quantization_matrixes)
            
        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

