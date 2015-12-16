import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt

import lasagne.updates
import lasagne.objectives

from utils.tasks import Duplicate
from utils.init import shared_glorot_uniform
from utils.cgru import CGRU
from utils.visualization import TaskAnimation


def model(input_var, target_var, num_symbols=4, \
    width=4, embedding_dim=24, filter_size=(3, 3)):

    # Input and Output weights
    E = shared_glorot_uniform(shape=(num_symbols, embedding_dim), name='E')
    O = shared_glorot_uniform(shape=(embedding_dim, num_symbols), name='O')
    # Layers
    layer1 = CGRU(filter_size=filter_size, width=width, \
        embedding=embedding_dim, name='layer1')

    def init_mental_image(i):
        embedded = T.dot(i, E)
        return embedded.dimshuffle(0, 3, 1, 2)

    def step(i_t, k_t, s_tm1):
        k_tp1 = k_t + 1
        s_t = layer1(s_tm1, k_t)
        return k_tp1, s_t

    s0 = init_mental_image(input_var)
    (_, mental_images), _ = theano.scan(step, \
        sequences=input_var.dimshuffle(2, 0, 1, 3), outputs_info=[0, s0])
    mental_images = T.concatenate([s0.dimshuffle('x', 0, 1, 2, 3), \
        mental_images], axis=0)

    def output_step(s_t):
        pre_logits = s_t.dimshuffle(0, 2, 1)
        logits = T.dot(pre_logits, O)
        output_t, _ = theano.map(T.nnet.softmax, \
            sequences=logits.dimshuffle(1, 0, 2))
        return output_t

    output = output_step(mental_images[-1,:,:,0,:])
    outputs, _ = theano.map(output_step,
                            sequences=mental_images[:,:,:,0,:])

    accuracy = T.mean(lasagne.objectives.categorical_accuracy(\
        output, target_var.dimshuffle(1, 0, 2)))
    cost = T.mean(T.nnet.categorical_crossentropy(\
        output, target_var.dimshuffle(1, 0, 2)))

    params = layer1.params + [E, O]

    return (mental_images.dimshuffle(1, 0, 3, 2, 4), \
        output.dimshuffle(1, 0, 2), \
        outputs.dimshuffle(2, 0, 1, 3), \
        accuracy, cost, params)


if __name__ == '__main__':
    input_var = T.dtensor4('input')
    target_var = T.dtensor3('target')
    batch_size = 128

    states, output, outputs, accuracy, cost, params = model(input_var, \
        target_var, num_symbols=3, width=4, embedding_dim=24, \
        filter_size=(3, 3))

    updates = lasagne.updates.adam(cost, params, learning_rate=1e-3)
    train_fn = theano.function([input_var, target_var], cost, updates=updates)
    accuracy_fn = theano.function([input_var, target_var], accuracy)

    generator = Duplicate(batch_size=batch_size, max_iter=1000, \
        proba_curriculum=0.2, max_length=10)

    try:
        for i, p, (example_input, example_output) in generator:
            batch_error = train_fn(example_input, example_output)
            batch_accuracy = accuracy_fn(example_input, example_output)
            print 'Batch %d, Error: %.6f, Accuracy: %.6f (Length: %d)' % \
                (i, batch_error, batch_accuracy, p['length'])
    except KeyboardInterrupt:
        pass

    mental_image_fn = theano.function([input_var], states)
    prediction_fn = theano.function([input_var], output)
    all_predictions_fn = theano.function([input_var], outputs)

    # Animation
    anim = TaskAnimation(generator=generator, mental_image_fn=mental_image_fn, \
        outputs_fn=all_predictions_fn, pad_frames=(10, 20), length=50, \
        interval=50, blit=False)
    plt.show()