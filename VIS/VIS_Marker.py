import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume, segment):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.segment = segment
    ax.index = volume.shape[0] // 2
    ax.imshow(mark_boundaries(volume[ax.index], segment[ax.index], color=(0.6, 0, 0)))
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.title(ax.index)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
    plt.title(ax.index)


def previous_slice(ax):
    volume = ax.volume
    segment = ax.segment
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(mark_boundaries(volume[ax.index], segment[ax.index], color=(0.6, 0, 0)))


def next_slice(ax):
    volume = ax.volume
    segment = ax.segment
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(mark_boundaries(volume[ax.index], segment[ax.index], color=(0.6, 0, 0)))

