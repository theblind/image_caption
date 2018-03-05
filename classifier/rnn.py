from builtins import range
from builtins import object
import numpy as np

from layers import *
from rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN.
        """
        (N, D) = features.shape
        (N, T) = captions.shape
        (W, H) = self.params['Wx'].shape
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        # Word embedding matrix
        W_embed = self.params['W_embed']
        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        # (1) Use an affine transformation to compute the initial hidden state
        # from the image features
        h0, hidden_cache = affine_forward(features, W_proj, b_proj)
        # (2) Use a word embedding layer to transform the words in captions_in
        # from indices to vectors
        word_vector, word_cache = word_embedding_forward(captions_in, W_embed)
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps
        if self.cell_type == "rnn":
          h, rnn_cache = rnn_forward(word_vector, h0, Wx, Wh, b)
        elif self.cell_type == "lstm":
          h, lstm_cache = lstm_forward(word_vector, h0, Wx, Wh, b)
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states
        out, affine_cache = temporal_affine_forward(h, W_vocab, b_vocab)
        # (5) Use (temporal) softmax to compute loss using captions_out
        loss, dout = temporal_softmax_loss(out, captions_out, mask, verbose=False)

        # Backward gradients
        dh, grads["W_vocab"], grads["b_vocab"] = temporal_affine_backward(dout, affine_cache)
        if self.cell_type == "rnn":
          dword, dh0, grads["Wx"], grads["Wh"], grads["b"] = rnn_backward(dh, rnn_cache)
        elif self.cell_type == "lstm":
          dword, dh0, grads["Wx"], grads["Wh"], grads["b"] = lstm_backward(dh, lstm_cache)
        grads["W_embed"] = word_embedding_backward(dword, word_cache)
        dx, grads["W_proj"], grads["b_proj"] = affine_backward(dh0, hidden_cache)

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        prev_c = 0
        prev_h, _ = affine_forward(features, W_proj, b_proj)
        prev_word = np.empty((N,), dtype=np.int32)
        prev_word.fill(self._start)
        for i in range(max_length):
          word_embedded = W_embed[prev_word]
          if self.cell_type == "rnn":
            prev_h, cache = rnn_step_forward(word_embedded, prev_h, Wx, Wh, b)
          elif self.cell_type == "lstm":
            prev_h, prev_c, cache = lstm_step_forward(word_embedded, prev_h, prev_c, Wx, Wh, b)
          words_score, _ = affine_forward(prev_h, W_vocab, b_vocab)
          captions[:, i] = np.argmax(words_score, axis=1)
          prev_word = captions[:, i]

        return captions
