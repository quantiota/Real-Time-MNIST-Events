"""
Visualize a single N-MNIST event recording.

Usage:
  python visualize.py                          # First sample of digit 0
  python visualize.py --digit 7 --index 3      # 4th sample of digit 7
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tonic
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='Visualize N-MNIST event recording')
    parser.add_argument('--digit', type=int, default=0, help='Digit class (0-9)')
    parser.add_argument('--index', type=int, default=0, help='Image index within digit class')
    args = parser.parse_args()

    dataset = tonic.datasets.NMNIST(save_to=SCRIPT_DIR, train=True)

    count = 0
    for idx in range(len(dataset)):
        events, label = dataset[idx]
        if label == args.digit:
            if count == args.index:
                break
            count += 1
    else:
        print(f"No image found for digit={args.digit}, index={args.index}")
        return

    x = events['x']
    y = events['y']
    t = events['t']
    p = events['p']

    t_ms = (t - t[0]) / 1000.0
    duration_ms = t_ms[-1]
    n_events = len(events)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'N-MNIST Digit {args.digit} (image #{args.index}) — {n_events} events, {duration_ms:.0f} ms',
                 fontsize=14)

    # 1. Accumulated frame (all events)
    ax = axes[0, 0]
    frame = np.zeros((35, 35))
    for i in range(n_events):
        frame[y[i], x[i]] += (2 * p[i] - 1)
    im = ax.imshow(frame, cmap='RdBu_r', origin='upper')
    ax.set_title('Accumulated frame (ON - OFF)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. Event density (how many events per pixel)
    ax = axes[0, 1]
    density = np.zeros((35, 35))
    for i in range(n_events):
        density[y[i], x[i]] += 1
    im = ax.imshow(density, cmap='hot', origin='upper')
    ax.set_title('Event density (count per pixel)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Three time slices
    ax = axes[0, 2]
    t_third = duration_ms / 3.0
    colors = np.zeros((35, 35, 3))
    for i in range(n_events):
        slice_idx = min(int(t_ms[i] / t_third), 2)
        colors[y[i], x[i], slice_idx] = min(colors[y[i], x[i], slice_idx] + 0.15, 1.0)
    ax.imshow(colors, origin='upper')
    ax.set_title('3 saccades (R=1st, G=2nd, B=3rd)')

    # 4. Polarity over time
    ax = axes[1, 0]
    polarity_signed = p.astype(float) * 2.0 - 1.0
    ax.scatter(t_ms, polarity_signed, s=1, alpha=0.3, c=['blue' if v > 0 else 'red' for v in polarity_signed])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Polarity')
    ax.set_title('Polarity stream')
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['-1 (OFF)', '+1 (ON)'])
    ax.grid(True, alpha=0.3)

    # 5. Event rate over time
    ax = axes[1, 1]
    bins = np.linspace(0, duration_ms, 50)
    ax.hist(t_ms, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Events per bin')
    ax.set_title('Event rate')
    ax.grid(True, alpha=0.3)

    # 6. Spatial trajectory (event positions colored by time)
    ax = axes[1, 2]
    sc = ax.scatter(x, y, c=t_ms, s=2, alpha=0.4, cmap='viridis')
    ax.set_xlim(-1, 35)
    ax.set_ylim(35, -1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Spatial trajectory (color = time)')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.046, label='ms')

    plt.tight_layout()
    outfile = os.path.join(SCRIPT_DIR, f'nmnist_digit{args.digit}_idx{args.index}.png')
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved: {outfile}")


if __name__ == '__main__':
    main()
