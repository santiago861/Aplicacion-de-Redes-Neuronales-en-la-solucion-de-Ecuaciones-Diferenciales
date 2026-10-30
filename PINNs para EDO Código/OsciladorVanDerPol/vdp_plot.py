import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = './results'
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)


def setup_matplotlib():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         11,
        'axes.labelsize':    12,
        'axes.titlesize':    12,
        'legend.fontsize':   10,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'figure.dpi':        120,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'lines.linewidth':   1.4,
        'axes.grid':         True,
        'grid.alpha':        0.3,
    })


def plot_trajectories(d):
    """Figura 1: trayectorias y1(t), y2(t) - PINN vs referencia."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    t = d['t_ref']

    axes[0].plot(t, d['y1_ref'],  '-',  color='k',       label='Referencia (Radau)')
    axes[0].plot(t, d['y1_pinn'], '--', color='tab:red', label='PINN')
    axes[0].set_ylabel(r'$y_1(t)$')
    axes[0].legend(loc='upper right')

    axes[1].plot(t, d['y2_ref'],  '-',  color='k',       label='Referencia (Radau)')
    axes[1].plot(t, d['y2_pinn'], '--', color='tab:red', label='PINN')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$y_2(t)$')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'vdp_trajectories.pdf'))
    plt.close(fig)


def plot_phase_portrait(d):
    """Figura 2: retrato de fase y1 vs y2."""
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot(d['y1_ref'],  d['y2_ref'],  '-',  color='k',
            label='Referencia (Radau)', alpha=0.9)
    ax.plot(d['y1_pinn'], d['y2_pinn'], '--', color='tab:red',
            label='PINN', alpha=0.9)
    ax.plot(d['y1_ref'][0], d['y2_ref'][0], 'o',
            color='tab:blue', markersize=8, label='Condicion inicial')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'vdp_phase.pdf'))
    plt.close(fig)


def plot_pointwise_error(d):
    """Figura 3: error puntual absoluto en el tiempo."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    t = d['t_ref']
    err1 = np.abs(d['y1_pinn'] - d['y1_ref'])
    err2 = np.abs(d['y2_pinn'] - d['y2_ref'])
    floor = 1e-12
    err1 = np.maximum(err1, floor)
    err2 = np.maximum(err2, floor)
    ax.semilogy(t, err1, '-', color='tab:blue',
                label=r'$|y_{1,\theta}-y_{1,\mathrm{ref}}|$')
    ax.semilogy(t, err2, '-', color='tab:orange',
                label=r'$|y_{2,\theta}-y_{2,\mathrm{ref}}|$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('Error puntual absoluto')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'vdp_error.pdf'))
    plt.close(fig)


def plot_loss(d):
    """Figura 4: evolución de la perdida durante el entrenamiento."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    it = d['hist_iter']
    loss = d['hist_loss']
    ax.semilogy(it, loss, '-', color='tab:purple', linewidth=1.2)

    if 'iter_phase_change' in d.files:
        ipc = int(d['iter_phase_change'])
        ax.axvline(ipc, color='gray', linestyle=':', linewidth=1.0,
                   label=f'Inicio L-BFGS (iter {ipc})')
        ax.legend(loc='upper right')

    ax.set_xlabel('Iteracion')
    ax.set_ylabel(r'$\mathcal{L}(\theta)$')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'vdp_loss.pdf'))
    plt.close(fig)


def main():
    setup_matplotlib()
    npz_path = os.path.join(RESULTS_DIR, 'base.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"No se encontro {npz_path}. Ejecuta primero vdp_train.py")
    d = np.load(npz_path)

    plot_trajectories(d)
    plot_phase_portrait(d)
    plot_pointwise_error(d)
    plot_loss(d)
    print(f"4 figuras PDF generadas en  {FIG_DIR}/")
    print("  - vdp_trajectories.pdf  ->  fig:vdp_trayectorias")
    print("  - vdp_phase.pdf         ->  fig:vdp_fase")
    print("  - vdp_error.pdf         ->  fig:vdp_error")
    print("  - vdp_loss.pdf          ->  fig:vdp_loss")


if __name__ == '__main__':
    main()
