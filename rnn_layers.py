from __future__ import print_function, division
from builtins import range
import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    h_raw = x.dot(Wx) + prev_h.dot(Wh) + b
    next_h = np.tanh(h_raw)
    cache = (h_raw, x, prev_h, Wx, Wh)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    (h_raw, x, prev_h, Wx, Wh) = cache
    dnext_h = (1 - np.tanh(h_raw) ** 2) * dnext_h

    dx = dnext_h.dot(Wx.T)
    dprev_h = dnext_h.dot(Wh.T)
    dWx = x.T.dot(dnext_h)
    dWh = prev_h.T.dot(dnext_h)
    db = dnext_h.sum(axis=0)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    (N, T, D) = x.shape
    (N, H) = h0.shape
    h = np.empty((N, T, H))
    cache_history = []

    prev_h = h0
    for i in range(T):
        prev_h, cache = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        h[:, i, :] = prev_h
        cache_history.append(cache)
    return h, cache_history


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    (N, T, H) = dh.shape
    (N, D) = cache[0][1].shape
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))

    dprev_h = 0
    for i in reversed(range(T)):
        dnext_h = dh[:, i, :] + dprev_h
        dx[:, i, :], dprev_h, dWx_step, dWh_step, db_step = rnn_step_backward(
            dnext_h, cache[i])
        dWx += dWx_step
        dWh += dWh_step
        db += db_step

    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out = W[x]
    cache = (x, W)
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings.

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    (x, W) = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    (N, H) = prev_h.shape
    activation_vector = x.dot(Wx) + prev_h.dot(Wh) + b
    i_gate = sigmoid(activation_vector[:, 0:H])
    f_gate = sigmoid(activation_vector[:, H:H*2])
    o_gate = sigmoid(activation_vector[:, H*2:H*3])
    g_gate = np.tanh(activation_vector[:, H*3:H*4])

    next_c = f_gate * prev_c + i_gate * g_gate
    next_h = o_gate * np.tanh(next_c)

    cache = (H, x, Wx, Wh,
             activation_vector, i_gate, f_gate, o_gate, g_gate,
             prev_c, prev_h, next_c, prev_h)
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    H, x, Wx, Wh, a, i, f, o, g, prev_c, prev_h, next_c, prev_h = cache
    do = dnext_h * np.tanh(next_c)
    dc = (1 - np.tanh(next_c) ** 2) * dnext_h * o
    dc += dnext_c
    df = dc * prev_c
    dprev_c = dc * f
    di = g * dc
    dg = i * dc
    a_i = a[:, 0 * H:1 * H]
    a_f = a[:, 1 * H:2 * H]
    a_o = a[:, 2 * H:3 * H]
    a_g = a[:, 3 * H:4 * H]
    da_i = di * (1 - sigmoid(a_i)) * sigmoid(a_i)
    da_f = df * (1 - sigmoid(a_f)) * sigmoid(a_f)
    da_o = do * (1 - sigmoid(a_o)) * sigmoid(a_o)
    da_g = dg * (1 - np.tanh(a_g) ** 2)
    da = np.hstack((da_i, da_f, da_o, da_g))
    dx = np.dot(da, Wx.T)
    dprev_h = np.dot(da, Wh.T)
    db = np.sum(da, axis=0, keepdims=False)
    dWx = np.dot(x.T, da)
    dWh = np.dot(prev_h.T, da)

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    (N, T, D) = x.shape
    (N, H) = h0.shape
    h = np.zeros((N, T, H))
    cache_history = []

    prev_h = h0
    prev_c = np.zeros((N, H))
    for i in range(T):
        data = x[:, i, :]
        prev_h, prev_c, cache = lstm_step_forward(
            data, prev_h, prev_c, Wx, Wh, b)
        h[:, i, :] = prev_h
        cache_history.append(cache)
    return h, cache_history


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    (N, T, H) = dh.shape
    (N, D) = cache[0][1].shape
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H*4))
    dWh = np.zeros((H, H*4))
    db = np.zeros((H*4, ))

    dh_step = 0
    dc_step = np.zeros([N, H])
    for i in reversed(range(T)):
        dh_step = dh[:, i, :] + dh_step
        dx_step, dh_step, dc_step, dWx_step, dWh_step, db_step = lstm_step_backward(
            dh_step, dc_step, cache[i])

        dx[:, i, :] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    dh0 = dh_step

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
