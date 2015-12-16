import theano
import theano.tensor as T
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

try:
    from palettable.cubehelix import jim_special_16
    default_cmap = jim_special_16.mpl_colormap
except ImportError:
    default_cmap = 'bone'

def set_xy_axis_visible(ax, visible=True):
    ax.get_xaxis().set_visible(visible)
    ax.get_yaxis().set_visible(visible)

def set_horizontal_markers(ax, width=4, embedding_dim=24, length=50, \
    style={'color': 'red'}):
    for i in xrange(1, width):
        ax.plot([-0.5, 2 * length - 0.5], [i * embedding_dim - 0.5, \
            i * embedding_dim - 0.5], **style)

def set_vertical_marker(ax, length=50, plot_height=2, style={'color': 'red'}):
    ax.plot([length - 0.5, length - 0.5], [-0.5, plot_height - 0.5], **style)

class TaskAnimation(animation.TimedAnimation):
    """docstring for TaskAnimation"""
    def __init__(self, generator, mental_image_fn, outputs_fn, \
        cmap=default_cmap, length=50, embedding_dim=24, pad_frames=(0, 0), \
        **kwargs):
        self.fig = plt.figure(figsize=(11, 9))
        self.generator = generator
        self.generator.batch_size = 1
        self.cmap = cmap
        self.length = length
        self.embedding_dim = embedding_dim
        self.pad_frames = pad_frames
        example_input, self.viz_input = self.sample()
        self.mental_images = mental_image_fn(example_input)
        self.outputs = outputs_fn(example_input)

        title_props = matplotlib.font_manager.FontProperties(\
            family='Helvetica', weight='bold', size=10)

        gs = gridspec.GridSpec(3, 2, width_ratios=[2 * length, 1],
                               height_ratios=[2, 4 * embedding_dim, 2])

        self.ax1 = plt.subplot(gs[0, 0])
        self.ax1.axis([-0.5, 2 * length - 0.5, -0.5, 1.5])
        set_xy_axis_visible(self.ax1, False)
        self.ax1.set_title('Input')
        self.ax1.title.set_font_properties(title_props)

        self.ax2 = plt.subplot(gs[1, 0])
        self.ax2.axis([-0.5, 2 * length - 0.5, -0.5, 4 * embedding_dim - 0.5])
        set_xy_axis_visible(self.ax2, False)
        self.ax2.set_title('Mental image')
        self.ax2.title.set_font_properties(title_props)

        self.ax3 = plt.subplot(gs[2, 0])
        self.ax3.axis([-0.5, 2 * length - 0.5, -0.5, 1.5])
        set_xy_axis_visible(self.ax3, False)
        self.ax3.set_title('Prediction')
        self.ax3.title.set_font_properties(title_props)

        self.ax4 = plt.subplot(gs[:, 1])
        set_xy_axis_visible(self.ax4, False)
        self.ax4.set_title('Time')
        self.ax4.title.set_font_properties(title_props)

        self.img1 = self.ax1.imshow(self.viz_input, interpolation='nearest', \
            cmap=cmap, aspect='auto', origin='lower')
        set_vertical_marker(self.ax1, length=length, plot_height=2, \
            style={'c': 'red', 'lw': 1})
        self.img2 = self.ax2.imshow(self.mental_images[0, 0].reshape((4 * \
            embedding_dim, 2 * length))[::-1], interpolation='nearest', \
            cmap=cmap, origin='lower', aspect='auto', vmin=-1., vmax=1.)
        set_vertical_marker(self.ax2, length=length, plot_height=4 * \
            embedding_dim, style={'c': 'red', 'lw': 1, 'ls': '-'})
        set_horizontal_markers(self.ax2, width=4, embedding_dim=embedding_dim, \
            length=length, style={'c': 'black', 'lw': 1})
        self.img3 = self.ax3.imshow(self.viz_output(0), \
            interpolation='nearest', cmap=cmap, aspect='auto')
        set_vertical_marker(self.ax3, length=length, plot_height=2, \
            style={'c': 'red', 'lw': 1})
        self.time = np.zeros((2 * length + 1, 1))
        self.time[0, 0] = 1
        self.img4 = self.ax4.imshow(self.time, interpolation='nearest', \
            cmap=cmap, aspect='auto')

        super(TaskAnimation, self).__init__(self.fig, **kwargs)

    def sample(self):
        example_input, example_output = \
            self.generator.sample(length=self.length)
        viz_input = np.zeros((2, 2 * self.length))
        viz_input[1, :] = example_input[0, 0, :, 0]
        viz_input[0, :] = example_input[0, 0, :, 2]
        return example_input, viz_input

    def viz_output(self, index):
        viz_output = np.zeros((2, 2 * self.length))
        viz_output[1, :] = self.outputs[0, index, :, 0]
        viz_output[0, :] = self.outputs[0, index, :, 2]
        return viz_output

    def new_frame_seq(self):
        return iter(np.arange(self.pad_frames[0] + self.pad_frames[1] + \
            2 * self.length + 1))

    def _draw_frame(self, framedata):
        self.time = np.zeros((2 * self.length + 1, 1))
        if framedata <= self.pad_frames[0]:
            index = 0
        elif framedata > self.pad_frames[0] + 2 * self.length:
            index = -1
        else:
            index = (framedata - self.pad_frames[0]) % (2 * self.length + 1)
        self.time[index, 0] = 1
        mental_image = self.mental_images[0, index]
        
        self.img1.set_data(self.viz_input)
        self.img2.set_data(mental_image.reshape((4 * self.embedding_dim, \
            2 * self.length))[::-1])
        self.img3.set_data(self.viz_output(index))
        self.img4.set_data(self.time)
        self._drawn_artists = [self.img1, self.img2, self.img3, self.img4]

    def _init_draw(self):
        self.img1.set_data([[]])
        self.img2.set_data([[]])
        self.img3.set_data([[]])
        self.img4.set_data([[]])
