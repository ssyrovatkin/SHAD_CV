import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            if updater.inertia is not None:
                updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            else:
                updater.inertia = self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        self.inputs = inputs
        return np.maximum(inputs, np.zeros_like(inputs))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        grad_inputs = np.multiply(grad_outputs, self.inputs >= 0)
        return grad_inputs
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        self.inputs = inputs
        max_inp = np.max(inputs, axis=1, keepdims=True)
        exp = np.exp(inputs - max_inp)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        self.output = (exp)/(sum_exp)
        return self.output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        local_derivative = np.zeros((grad_outputs.shape[0], grad_outputs.shape[1], grad_outputs.shape[1]))
        local_softmax = self.output

        for i in range(grad_outputs.shape[0]):
            local_derivative[i] = np.diag(local_softmax[i])
            local_derivative[i] -= local_softmax[i].reshape(-1, 1) @ local_softmax[i].reshape(-1, 1).T

        self.grad_input = np.matmul(grad_outputs[:, None, :], local_derivative)[:, 0, :]
        return self.grad_input
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.inputs = inputs
        self.outputs = inputs @ self.weights + self.biases

        return self.outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.grad_inputs = grad_outputs @ self.weights.T

        self.weights_grad = self.inputs.T @ grad_outputs
        self.biases_grad = np.sum(grad_outputs, axis=0)

        return self.grad_inputs
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        self.pred = y_pred
        self.target = y_gt
        m = y_gt.shape[0]

        self.output = -np.sum(np.log(np.sum(y_pred * y_gt, axis=-1))) / m

        return self.output.reshape(1,)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        y = np.maximum(np.ones_like(y_pred) * eps, y_pred)
        m = y_gt.shape[0]
        self.grad_inputs = -y_gt / y

        return self.grad_inputs / m
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGD(lr=0.001))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(units=256, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(units=128))
    model.add(ReLU())
    model.add(Dense(units=64))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(ReLU())
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=8, epochs=5)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def _pad_zeros(tensor, one_side_pad, axis=[-1, -2]):
    """
    Добавляет одинаковый паддинг по осям, указанным в axis.
    Метод не проверяется в тестах -- можно релизовать слой без
    использования этого метода.
    """
    N, C, H, W = tensor.shape

    target_shape = [N, C, H, W]
    for a in axis:
        target_shape[a] += 2 * one_side_pad

    for dim_in, dim_target in zip(tensor.shape, target_shape):
        assert dim_target >= dim_in

    pad_width = []
    for dim_in, dim_target in zip(tensor.shape, target_shape):
        if (dim_in - dim_target) % 2 == 0:
            pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
        else:
            pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

    return np.pad(tensor, pad_width, 'constant', constant_values=0)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    N, _, H, W = inputs.shape
    c, d, kh, kw = kernels.shape

    H = 1 + int(H + 2 * padding - kh)
    W = 1 + int(W + 2 * padding - kw)
    flip_kernels = np.expand_dims(np.flip(kernels, axis=(2,3)), axis=0)

    if padding != 0:
        xpad = _pad_zeros(inputs, padding)
    else:
        xpad = inputs
    output = np.zeros((N, c, H, W))

    for i in range(H):
        h_start = i
        h_end = i + kh
        for j in range(W):
            w_start = j
            w_end = j + kw
            output[:, :, i, j] = np.sum(
                xpad[:, None, :, h_start: h_end, w_start:w_end] * flip_kernels, axis=(2,3,4))

    del xpad

    return output
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        self.inputs = inputs
        C_out, C_in, H, W = self.kernels.shape
        self.padding = (self.kernel_size - 1) // 2
        output = convolve_numpy(inputs, self.kernels, self.padding) + self.biases.reshape(1,C_out,1,1)

        return output

        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        self.biases_grad = np.sum(grad_outputs, axis=(0,2,3))
        self.kernels_grad = convolve_numpy(np.transpose(np.flip(self.inputs, (2,3)), (1,0,2,3)), np.transpose(grad_outputs, (1,0,2,3)), self.padding)
        self.kernels_grad = np.transpose(self.kernels_grad, (1,0,2,3))
        padding = self.kernel_size - self.padding - 1
        self.grad_inputs = convolve_numpy(grad_outputs, np.transpose(np.flip(self.kernels, (2,3)), (1,0,2,3)), padding)

        return self.grad_inputs
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        N, C, H, W = inputs.shape

        H = 1 + int((H - self.pool_size) // self.pool_size)
        W = 1 + int((W - self.pool_size) // self.pool_size)

        self.inputs = inputs
        self.outputs = np.zeros((N, C, H, W))
        self.mask = np.zeros((N, C, H, W))

        for i in range(H):
            h_start = i * self.pool_size
            h_end = i * self.pool_size + self.pool_size
            for j in range(W):
                w_start = j * self.pool_size
                w_end = j * self.pool_size + self.pool_size
                if self.pool_mode == 'max':
                    self.outputs[:, :, i, j] = np.max(inputs[:, :, h_start: h_end, w_start:w_end], axis=(2,3))
                elif self.pool_mode == 'avg':
                    self.outputs[:, :, i, j] = np.mean(inputs[:, :, h_start: h_end, w_start:w_end], axis=(2,3))

        return self.outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        N, C, H, W = self.inputs.shape
        _, _, Hout, Wout = grad_outputs.shape

        self.input_grad = np.zeros(self.inputs.shape)

        if self.pool_mode == 'avg':
            for i in range(Hout):
                h_start = i * self.pool_size
                h_end = i * self.pool_size + self.pool_size
                for j in range(Wout):
                    w_start = j * self.pool_size
                    w_end = j * self.pool_size + self.pool_size
                    g = grad_outputs[:, :, i, j] / (self.pool_size ** 2)
                    self.input_grad[:, :, h_start: h_end, w_start:w_end] += g[:,:,None,None]
        elif self.pool_mode == 'max':
            for xn in range(N):
                for fn in range(C):
                    for i in range(Hout):
                        h_start = i * self.pool_size
                        h_end = i * self.pool_size + self.pool_size
                        for j in range(Wout):
                            w_start = j * self.pool_size
                            w_end = j * self.pool_size + self.pool_size
                            window = self.inputs[xn, fn, h_start: h_end, w_start:w_end]
                            max_ids = np.unravel_index(window.argmax(), window.shape)
                            self.input_grad[xn, fn, h_start: h_end, w_start:w_end][max_ids] = grad_outputs[xn, fn, i, j]

        return self.input_grad
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        self.inputs = inputs

        m, c, h, w = inputs.shape
        N = m * h * w
        self.X_mean = np.sum(inputs, axis=(0, 2, 3), keepdims=True)/N
        self.X_var = np.sum((inputs-self.X_mean)**2, axis=(0, 2, 3), keepdims=True)/N

        if self.is_training:
            self.running_mean = self.X_mean*(1-self.momentum) + self.running_mean.reshape(1,c,1,1)*self.momentum
            self.running_var = self.X_var*(1-self.momentum) + self.running_var.reshape(1,c,1,1)*self.momentum

            self.xhat = np.divide((inputs - self.X_mean), np.sqrt(self.X_var + eps))
        else:
            self.xhat = np.divide((inputs - self.running_mean.reshape(1,c,1,1)), np.sqrt(self.running_var.reshape(1,c,1,1) + eps))

        self.outputs = self.gamma.reshape(1,c,1,1) * self.xhat + self.beta.reshape(1,c,1,1)
        return self.outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        m, c, h, w = self.inputs.shape
        Nt = m * h * w

        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        self.gamma_grad = np.sum(grad_outputs * self.xhat, axis=(0, 2, 3))

        self.dxhat = grad_outputs * self.gamma.reshape(1, c, 1, 1)
        dsigma = np.sum(self.dxhat * (self.inputs - self.X_mean), axis=(0, 2, 3)).reshape(1,c,1,1) * -0.5 * (self.X_var + eps) ** -1.5
        dmu = np.sum(self.dxhat * (-1.0 / np.sqrt(self.X_var + eps)), axis=(0, 2, 3)).reshape(1,c,1,1) + \
              dsigma * np.sum(-2 * (self.inputs - self.X_mean), axis=(0, 2, 3)).reshape(1,c,1,1) / Nt

        self.dx = self.dxhat * (1.0 / np.sqrt(self.X_var + eps)) + dsigma * (
                    2.0 * (self.inputs - self.X_mean)) / Nt + dmu / Nt
        return self.dx
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        self.inputs = inputs
        self.outputs = inputs.reshape(len(inputs), -1)
        return self.outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        self.grad_inputs = grad_outputs.reshape(self.inputs.shape)
        return self.grad_inputs
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.uniform(0, 1, size=inputs.shape) > self.p
            self.output = inputs*self.forward_mask
        else:
            self.output = inputs * (1 - self.p)

        return  self.output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        self.grad_inputs = grad_outputs*self.forward_mask
        return self.grad_inputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGDMomentum(lr=0.1, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(output_channels=12, kernel_size=3, input_shape=(3,32,32)))
    #model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode='avg'))
    model.add(Conv2D(output_channels=32, kernel_size=3))
    #model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2, pool_mode='avg'))
    #model.add(Conv2D(output_channels=32, kernel_size=3))
    #model.add(BatchNorm())
    #model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=256, epochs=5)

    # your code here /\
    return model

# ============================================================================
