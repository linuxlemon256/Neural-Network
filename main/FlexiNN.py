import numpy as xp

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = xp.dot(self.x, self.W) + self.b
        return out
    def backward(self, dout):
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        dx = xp.dot(dout, self.W.T)
        dx = dx.reshape(self.original_x_shape)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        x = xp.clip(x, -500, 500)
        out = 1.0 / (1.0 + xp.exp(xp.clip(-x, -500, 500)))
        self.out = out
        return out
    def backward(self, dout):
        dx = self.out*(1.0-self.out)*dout
        return dx

class Relu:
    def __init__(self):
        self.x = None
    def forward(self,x):
        self.x = x
        out = xp.maximum(0, self.x)
        return out
    def backward(self, dout):
        dx = dout * xp.where(self.x > 0, 1, 0)
        return dx

class LeakyRelu:
    def __init__(self, alpha=0.1):
        self.x = None
        self.alpha = alpha
    def forward(self,x):
        self.x = x
        out = xp.where(x > 0, x, x * self.alpha)
        return out
    def backward(self, dout):
        dx = dout * xp.where(self.x > 0, 1, self.alpha)
        return dx

class Tanh:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = xp.tanh(x)            #out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * (1 - self.out**2)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.out = None
        self.t = None
        self.batch = None
        self.loss_date = None
    def softmax(self,x):
        x -= xp.max(x, axis=-1, keepdims=True)
        exp_x = xp.exp(x)
        out = exp_x / xp.sum(exp_x, axis=-1, keepdims=True)
        self.out = out
        return out
    def loss(self,x):
        eps = 1e-12
        self.batch = x.shape[0]
        x = xp.clip(x, eps, 1 - eps)
        out = -xp.sum(self.t * xp.log(x)) / x.shape[0]
        self.loss_date = out
        return out
    def forward(self,x,t):
        self.t = t
        out = self.softmax(x)
        out = self.loss(out)
        return out
    def backward(self, dout):
        dx = dout*(self.out - self.t)/self.batch
        return dx

def im2col(x, FH, FW, stride = 1, pad = 0):
    # 鎶婅緭鍏ュ睍寮€鎴愬垪鐭╅樀锛氭瘡涓€鍒楁槸涓€涓獥鍙?
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1
    img = xp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")
    col = xp.zeros((N, C, FH, FW, out_h, out_w))
    for y in range(FH):
        y_max = y + stride * out_h
        for xx in range(FW):
            xx_max = xx + stride * out_w
            col[:, :, y, xx, :, :] = img[:, :, y:y_max:stride, xx:xx_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, x_shape, FH, FW, stride = 1, pad = 0):
    # im2col 鐨勯€嗘搷浣滐細鎶婂垪鐭╅樀鐨勬搴﹀姞鍥炶緭鍏?
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1
    col = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
    img = xp.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for y in range(FH):
        y_max = y + stride * out_h
        for xx in range(FW):
            xx_max = xx + stride * out_w
            img[:, :, y:y_max:stride, xx:xx_max:stride] += col[:, :, y, xx, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]


class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.FN, self.C, self.FH, self.FW = W.shape
        self.x = None
        self.col = None
    def forward(self,x):
        self.x = x
        N, C, H, W = x.shape
        OH = (H + 2 * self.pad - self.FH) // self.stride + 1
        OW = (W + 2 * self.pad - self.FW) // self.stride + 1
        col = im2col(x, self.FH, self.FW, self.stride, self.pad)
        self.col = col
        out = xp.dot(col, self.W.reshape(self.FN, -1).T)
        out = out.reshape(N, OH, OW, self.FN).transpose(0, 3, 1, 2)
        out = out + self.b.reshape(1, self.FN, 1, 1)
        return out
    def backward(self, dout):
        N, C, H, W = self.x.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.FN)
        dw = xp.dot(self.col.T, dout)
        dw = dw.reshape(self.C, self.FH, self.FW, self.FN).transpose(3, 0, 1, 2)
        db = xp.sum(dout, axis = 0)
        dcol = xp.dot(dout, self.W.reshape(self.FN, -1))
        dx = col2im(dcol, self.x.shape, self.FH, self.FW, self.stride, self.pad)
        self.dW = dw
        self.db = db
        return dx

class Pooling:
    def __init__(self, pool_h = 2, pool_w = 2, methods = "max"):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.methods = "max"
        if methods == "average":
            self.methods = "average"
    def forward(self,x):
        N, C, H, W = x.shape
        OH = H // self.pool_h
        OW = W // self.pool_w
        xr = x.reshape(N, C, OH, self.pool_h, OW, self.pool_w)
        if self.methods == "max":
            out = xr.max(axis = (3, 5))
            self.mask = (xr == out[:, :, :, None, :, None])
        elif self.methods == "average":
            out = xp.mean(xr, axis = (3, 5))
            self.mask = (xr == out[:, :, :, None, :, None])
        return out
    def backward(self,dout):
        dout_r = dout.reshape(dout.shape[0], dout.shape[1], dout.shape[2], 1, dout.shape[3], 1)
        if self.methods == "average":
            dx = xp.broadcast_to(dout_r / (self.pool_h * self.pool_w), self.mask.shape).reshape(self.mask.shape[0], self.mask.shape[1], self.mask.shape[2] * self.pool_h, self.mask.shape[4] * self.pool_w)
        else:
            dx = (self.mask * dout_r).reshape(self.mask.shape[0], self.mask.shape[1], self.mask.shape[2] * self.pool_h, self.mask.shape[4] * self.pool_w)
        return dx

class Dropout:
    def __init__(self,rate = 0):
        self.rate = rate
        self.mask = None
    def forward(self,x):
        self.mask = xp.random.rand(*x.shape) > self.rate
        return xp.multiply(x, self.mask)
    def backward(self,dout):
        return xp.multiply(dout, self.mask)
    def predict(self,x):
        return xp.multiply(x, 1.0 - self.rate)

class MLP:
    def __init__(self,input_size,
                 hidden_size,
                 output_size,
                 learning_rate = 0.1,
                 learning_time = 1000,
                 init_weights = None,
                 print_every = 100,
                 print_output = True,
                 activation = "leaky_relu",
                 backpropagation = "error-back",
                 decay_rate = None,
                 dropout_rate = 0.0,
                 ):

        self.input_size = input_size
        self.hidden_size = hidden_size if type(hidden_size) == list else [hidden_size]
        self.output_size = output_size
        self.net_layer = len(self.hidden_size)+1
        self.learning_rate = learning_rate
        self.learning_time = learning_time
        self.activation = activation
        self.backpropagation = backpropagation
        self.print_every = print_every
        self.print_output = print_output
        self.params = {}
        self.decay_rate = decay_rate
        self.dropout_rate = dropout_rate


        if init_weights is None:
            if self.activation in ("leaky_relu", "relu"):
                weights = 2.0
            elif self.activation in ("sigmoid", "tanh"):
                weights = 1.0
        else:
            weights = init_weights


        for layer in range(self.net_layer):
            w_layer = "W" + str(layer)
            b_layer = "b" + str(layer)
            if layer == 0:
                self.params[w_layer] = xp.random.randn(self.input_size, self.hidden_size[0]).astype(xp.float32) * xp.sqrt(weights / self.input_size)
                self.params[b_layer] = xp.zeros(self.hidden_size[0]).astype(xp.float32)
            elif layer < self.net_layer-1:
                self.params[w_layer] = xp.random.randn(self.hidden_size[layer - 1], self.hidden_size[layer]).astype(xp.float32) * xp.sqrt(weights / self.hidden_size[layer - 1])
                self.params[b_layer] = xp.zeros(self.hidden_size[layer]).astype(xp.float32)
            elif layer == self.net_layer-1:
                self.params[w_layer] = xp.random.randn(self.hidden_size[-1], self.output_size).astype(xp.float32) * xp.sqrt(weights / self.hidden_size[-1])
                self.params[b_layer] = xp.zeros(self.output_size).astype(xp.float32)



    def train(self,x, t,
              learning_time = None,
              learning_rate = None,
              batch_size = None,
              epochs = 100,
              decay_rate = None,
              dropout_rate = None,
              print_every = None,
              print_output = None):


        if learning_time is not None:
            self.learning_time = learning_time
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if print_every is not None:
            self.print_every = print_every
        if print_output is not None:
            self.print_output = print_output
        if decay_rate is not None:
            self.decay_rate = decay_rate
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate


        self.t = t
        learning_rate = self.learning_rate
        softmax_with_loss = SoftmaxWithLoss()
        if batch_size is None:
            for lt in range(self.learning_time):
                out = x
                d = self.forward(out,softmax_with_loss)
                grads= self.backward(d,softmax_with_loss)
                self.params = self.update(grads)
                if self.print_output is True:
                    time = lt + 1
                    if time % self.print_every == 0:
                        print(f"Iteration : {time} / {self.learning_time}\t|\tloss : {softmax_with_loss.loss_date:.6f}\t")
        else:
            for epoch in range(epochs):
                n = xp.random.permutation(x.shape[0])
                x_shuffle = x[n]
                t_shuffle = t[n] if t.ndim == 1 else t[n]
                for start in range(0, x.shape[0], batch_size):
                    end = min(start + batch_size,x.shape[0])
                    train = x_shuffle[start:end]
                    test = t_shuffle[start:end]
                    self.t = test
                    d = self.forward(train, softmax_with_loss)
                    grads = self.backward(d,softmax_with_loss)
                    self.params = self.update(grads)
                if (epoch+1)%self.print_every == 0 and self.print_output:
                    print(f"Epoch: {epoch+1}/{epochs}\t{(epoch+1)*100/epochs:.2f}%\t|\tloss : {softmax_with_loss.loss_date:.6f}\t")
        self.learning_rate = learning_rate


    def predict(self,x):
        activation = self.load_activation()
        softmax_with_loss = SoftmaxWithLoss()
        dropout = Dropout(self.dropout_rate)
        out = x
        for layer in range(self.net_layer):
            W_layer = self.params["W" + str(layer)]
            b_layer = self.params["b" + str(layer)]
            affine = Affine(W_layer, b_layer)
            if layer < self.net_layer-1:
                out = affine.forward(out)
                out = activation.forward(out)
                out = dropout.predict(out)
            else:
                out = affine.forward(out)
                out = softmax_with_loss.softmax(out)
        return out


    def accuracy(self,x,t):
        y_pred = xp.argmax(x, axis=1)  # 棰勬祴绫诲埆
        y_true = xp.argmax(t, axis=1)  # 鐪熷疄绫诲埆
        acc = xp.mean(y_pred == y_true)  # 璁＄畻姝ｇ‘姣斾緥
        return acc


    def load_activation(self):
        activation = None
        if self.activation == "leaky_relu":
            activation = LeakyRelu()
        elif self.activation == "sigmoid":
            activation = Sigmoid()
        elif self.activation == "tanh":
            activation = Tanh()
        elif self.activation == "relu":
            activation = Relu()
        return activation

    def backward(self,d,softmax_with_loss):
        grads = {}
        dout = 1
        for layer in reversed(range(self.net_layer)):
            affine, activation, dropout = d[layer]
            if layer == self.net_layer - 1:
                dout = softmax_with_loss.backward(dout)
            else:
                dout = dropout.backward(dout)
                dout = activation.backward(dout)
            dout = affine.backward(dout)
            grads["W" + str(layer)] = affine.dW
            grads["b" + str(layer)] = affine.db
        return grads

    def forward(self,out,softmax_with_loss):
        d={}
        for layer in range(self.net_layer):
            W_layer = self.params["W" + str(layer)]
            b_layer = self.params["b" + str(layer)]
            affine = Affine(W_layer, b_layer)

            if layer < self.net_layer - 1:
                dropout = Dropout(self.dropout_rate)
                activation = self.load_activation()
                out = affine.forward(out)
                out = activation.forward(out)
                out = dropout.forward(out)
                d[layer] = (affine, activation, dropout)
            else:
                out = affine.forward(out)
                out = softmax_with_loss.forward(out, self.t)
                d[layer] = (affine, None, None)
        return d

    def update(self,grads):
        params = self.params
        for layer in range(self.net_layer):
            params["W" + str(layer)] -= self.learning_rate * grads["W" + str(layer)]
            params["b" + str(layer)] -= self.learning_rate * grads["b" + str(layer)]
        if self.decay_rate:
            self.learning_rate *= self.decay_rate
        return params


    def save(self,name = None):
        if name is None:
            name = "_.npz"
        else:
            name = str(name)
        if name[-4:] != ".npz":
            name += ".npz"
        xp.savez_compressed(name, **self.params)

    def load(self,name = None):
        if name is None:
            name = "_.npz"
        else:
            name = str(name)
        if name[-4:] != ".npz":
            name += ".npz"
        params=xp.load(name)
        params_dict = {}
        try:
            layer = 0
            while True:
                params_dict["W" + str(layer)] = params["W" + str(layer)]
                params_dict["b" + str(layer)] = params["b" + str(layer)]
                layer +=1
        except KeyError:
            print("rebuilding...")
            self.net_layer = layer
            print("done")
            print(f"layer: {layer}\t|\tinput shape: {self.input_size}\t|\thidden shape: {self.hidden_size}\t|\toutput shape: {self.output_size}")
        self.params = params_dict

class CNN:
    def __init__(self,input_shape,output_size,conv_w,conv_b,pool_h,pool_w,activation = "relu",
                 learning_rate = 0.1,
                 learning_time = 1000,
                 print_every = 100,
                 print_output = True,
                 ):
        self.input_shape = input_shape
        self.output_size = output_size
        self.conv_w = conv_w
        self.conv_b = conv_b
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.activation = activation
        self.learning_rate = learning_rate
        self.learning_time = learning_time
        self.print_every = print_every
        self.print_output = print_output
        self.conv = Convolution(W=self.conv_w, b=self.conv_b, )
        self.act = self.load_activation()
        self.pool = Pooling(pool_h=self.pool_h, pool_w=self.pool_w)

        C, H, W = input_shape
        FH, FW = conv_w.shape[2], conv_w.shape[3]
        conv_H = H - FH + 1
        conv_W = W - FW + 1
        pool_H = int(conv_H / pool_h)
        pool_W = int(conv_W / pool_w)
        flattened = conv_w.shape[0] * pool_H * pool_W
        self.affine = Affine(
            W = xp.random.randn(flattened, output_size) *0.1,
            b = xp.zeros(output_size)
        )
    def forward(self,x,softmax_with_loss = None):

        out = self.conv.forward(x)
        out = self.act.forward(out)
        out = self.pool.forward(out)
        self.pool_out_shape = out.shape
        N = out.shape[0]
        out = out.reshape(N, -1)
        out = self.affine.forward(out)
        if softmax_with_loss is not None:
            out = softmax_with_loss.forward(out, self.t)
        return out

    def backward(self,softmax_with_loss):
        dout = softmax_with_loss.backward(1)
        dout = self.affine.backward(dout)
        dout = dout.reshape(self.pool_out_shape)
        dout = self.pool.backward(dout)
        dout = self.act.backward(dout)
        dout = self.conv.backward(dout)
        return dout

    def update(self):
        self.conv.W -= self.learning_rate * self.conv.dW
        self.conv.b -= self.learning_rate * self.conv.db
        self.affine.W -= self.learning_rate * self.affine.dW
        self.affine.b -= self.learning_rate * self.affine.db

    def train(self,x, t,
              learning_time = None,
              learning_rate = None,
              print_every = None,
              print_output = None):
        if learning_time is not None:
            self.learning_time = learning_time
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if print_every is not None:
            self.print_every = print_every
        if print_output is not None:
            self.print_output = print_output
        self.t = t
        softmax_with_loss = SoftmaxWithLoss()
        for lt in range(self.learning_time):
            self.forward(x, softmax_with_loss)
            self.backward(softmax_with_loss)
            self.update()
            if self.print_output is True:
                time = lt + 1
                if time % self.print_every == 0:
                    print(f"Iteration : {time} / {self.learning_time}\t|\tloss : {softmax_with_loss.loss_date:.6f}\t")

    def accuracy(self,x,t):
        y_pred = xp.argmax(x, axis=1)
        y_true = xp.argmax(t, axis=1)
        acc = xp.mean(y_pred == y_true)
        return acc

    def load_activation(self):
        activation = None
        if self.activation == "leaky_relu":
            activation = LeakyRelu()
        elif self.activation == "sigmoid":
            activation = Sigmoid()
        elif self.activation == "tanh":
            activation = Tanh()
        elif self.activation == "relu":
            activation = Relu()
        return activation

    def predict(self, x):
        return xp.argmax(self.forward(x), axis=1)
