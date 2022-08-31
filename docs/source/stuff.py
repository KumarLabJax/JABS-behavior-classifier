import numpy as np
import pandas as pd
import pathlib
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib

def load_file(filename):
    with h5py.File(filename, 'r') as fh:
        group = fh['predictions']
        predicted_class = group['predicted_class'][()]
        predicted_prob = group['probabilities'][()]

        df = pd.DataFrame(np.column_stack((predicted_class.flatten(), predicted_prob.flatten())),columns=['Class', 'Probability'])

        df = df[df['Class']!=-1]
        df['frame_no'] = df.index

    return df



def draw_ethogram(df, i, save_fig=False):

    fig = plt.figure(figsize=(12,1), dpi=150)
    ax = fig.add_subplot()

    cmap = matplotlib.colors.ListedColormap(['white', 'darkred'])


    ax.imshow(df.reshape(1,-1), interpolation=None, aspect='auto', cmap=cmap)
 
    
    s_frame = 0 + 1000*i
    e_frame = 1000 + 1000*i
    formatter = FuncFormatter(lambda x_val, tick_pos: f"{int((x_val-s_frame))}")

  
    ax.set_xlim(s_frame, e_frame)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(formatter)

    if save_fig:
        plt.savefig(f'figures/Ethogram_{i}.pdf', dpi=300, bbox_inches='tight', format='pdf')

    plt.show() 