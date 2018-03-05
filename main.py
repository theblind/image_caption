# As usual, a bit of setup
from __future__ import print_function

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from captioning_solver import CaptioningSolver, evaluate_model
from classifier.rnn import CaptioningRNN
from coco_utils import decode_captions, load_coco_data, sample_coco_minibatch
from image_utils import image_from_url


def main():
    # Load COCO data from disk
    data = load_coco_data()
    # Create Caption Model
    model = CaptioningRNN(
        cell_type='lstm',
        word_to_idx=data['word_to_idx'],
        input_dim=data['train_features'].shape[1],
        hidden_dim=512,
        wordvec_dim=256,
        dtype=np.float32,
    )
    solver = CaptioningSolver(model, data,
                              update_rule='adam',
                              num_epochs=1,
                              batch_size=100,
                              optim_config={
                                  'learning_rate': 5e-3,
                              },
                              lr_decay=0.995,
                              verbose=True, print_every=10, eval_every=100
                              )
    solver.train()

    # Plot the training losses
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    plt.plot(solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()

    # # Evaluate the model
    # evaluate_model(model, data)


if __name__ == "__main__":
    main()
