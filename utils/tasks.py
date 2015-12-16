import numpy as np

class Task(object):
    """docstring for Task"""
    def __init__(self, max_iter=None, proba_curriculum=0.2, batch_size=10, \
        width=4):
        self.max_iter = max_iter
        self.num_iter = 0
        self.proba_curriculum = proba_curriculum
        self.batch_size = batch_size
        self.width = width

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            u = np.random.rand()
            if u < self.proba_curriculum:
                length = np.random.randint(1, 21)
                params = self.sample_params(length=length)
            else:
                params = self.sample_params()
            self.num_iter += 1
            return (self.num_iter - 1), params, self.sample(**params)
        else:
            raise StopIteration()


class BinaryAdd(Task):
    """docstring for BinaryAdd"""
    def __init__(self, min_length=1, max_length=6, \
        max_iter=None, batch_size=10, width=4, proba_curriculum=0.2):
        super(BinaryAdd, self).__init__(max_iter=max_iter, \
            batch_size=batch_size, width=width, \
            proba_curriculum=proba_curriculum)
        self.min_length = min_length
        self.max_length = max_length

    def sample_params(self, length=None):
        if length is None:
            length = np.random.randint(self.min_length, \
                self.max_length + 1)
        return {'length': length}

    def sample(self, length):
        power_two = 1 << np.arange(length)
        input_A = np.random.binomial(1, 0.5, (self.batch_size, length))
        input_B = np.random.binomial(1, 0.5, (self.batch_size, length))
        output = np.dot(input_A, power_two) + np.dot(input_B, power_two)
        example_input = np.zeros((self.batch_size, self.width, \
            2 * length + 1, 4))
        example_output = np.zeros((self.batch_size, \
            2 * length + 1, 4))

        example_input[:, 0, :length, 0] = input_A
        example_input[:, 0, :length, 1] = 1 - input_A
        example_input[:, 0, length, 2] = 1
        example_input[:, 0, -length:, 0] = input_B
        example_input[:, 0, -length:, 1] = 1 - input_B

        # Fill other tapes with blank symbols
        for i in xrange(1, self.width):
            example_input[:, i, :, 3] = 1

        for i in xrange(length + 1):
            bits = output % 2
            example_output[:, i, 0] = bits
            example_output[:, i, 1] = 1 - bits
            output >>= 1
        example_output[:, length+1:, 3] = 1

        return example_input, example_output


class Duplicate(Task):
    """docstring for Duplicate"""
    def __init__(self, min_length=1, max_length=6, \
        max_iter=None, batch_size=10, width=4, proba_curriculum=0.2):
        super(Duplicate, self).__init__(max_iter=max_iter, \
            batch_size=batch_size, width=width, \
            proba_curriculum=proba_curriculum)
        self.min_length = min_length
        self.max_length = max_length

    def sample_params(self, length=None):
        if length is None:
            length = np.random.randint(self.min_length, \
                self.max_length + 1)
        return {'length': length}

    def sample(self, length):
        sequence = np.random.binomial(1, 0.5, (self.batch_size, length))
        example_input = np.zeros((self.batch_size, self.width, \
            2 * length, 3))
        example_output = np.zeros((self.batch_size, \
            2 * length, 3))

        example_input[:, 0, :length, 0] = sequence
        example_input[:, 0, :length, 1] = 1 - sequence
        example_input[:, 0, length:, 2] = 1

        example_output[:, :length, 0] = sequence
        example_output[:, :length, 1] = 1 - sequence
        example_output[:, length:, 0] = sequence
        example_output[:, length:, 1] = 1 - sequence
        
        # Fill other tapes with blank symbols
        for i in xrange(1, self.width):
            example_input[:, i, :, 2] = 1

        return example_input, example_output