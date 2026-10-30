import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

DATA_FILE = './results/base_heat.npz'
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontro {DATA_FILE}. Ejecuta heat_train.py primero.")

    d = np.load(DATA_FILE)
    x_eval = d['x_eval']
    t_eval = d['t_eval']
    X = d['X_grid']
    T = d['T_grid']
    U_pinn = d['U_pinn']
    U_ref = d['U_ref']
    alpha = float(d['alpha'])
    err = np.abs(U_pinn - U_ref)

    hist_iter   = d['hist_iter']
    hist_loss   = d['hist_loss']
    hist_L_r    = d['hist_L_r']
    hist_L_ic   = d['hist_L_ic']
    hist_L_bc   = d['hist_L_bc']
    iter_change = int(d['iter_phase_change'])

    # ----------------------------------------------------------
    # Figura 1: 3 paneles (PINN, referencia, |error|)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(U_pinn.max(), U_ref.max())
    cmap_uv = plt.cm.hot_r

    im0 = axes[0].pcolormesh(T, X, U_pinn, shading='auto',
                              cmap=cmap_uv, vmin=0, vmax=vmax)
    axes[0].set_title(r'PINN: $u_\theta(x,t)$')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$x$')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(T, X, U_ref, shading='auto',
                              cmap=cmap_uv, vmin=0, vmax=vmax)
    axes[1].set_title(rf'Referencia: $\sin(\pi x)\,e^{{-{alpha}\pi^2 t}}$')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$x$')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(T, X, err, shading='auto', cmap='viridis')
    axes[2].set_title(r'Error puntual $|u_\theta - u_{\rm ref}|$')
    axes[2].set_xlabel(r'$t$')
    axes[2].set_ylabel(r'$x$')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'heat_solution.pdf'))
    plt.close(fig)
    print('Generado: heat_solution.pdf')

    # ----------------------------------------------------------
    # Figura 2: cortes temporales
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    t_snapshots = [0.0, 0.25, 0.5, 1.0]
    for ax, t_val in zip(axes.flatten(), t_snapshots):
        idx = int(round(t_val / (t_eval[1] - t_eval[0])))
        idx = min(idx, len(t_eval) - 1)
        ax.plot(x_eval, U_ref[:, idx], 'k-',  lw=2,   label='Referencia')
        ax.plot(x_eval, U_pinn[:, idx], 'r--', lw=1.5, label='PINN')
        ax.set_title(f'$t = {t_eval[idx]:.2f}$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u(x, t)$')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'heat_snapshots.pdf'))
    plt.close(fig)
    print('Generado: heat_snapshots.pdf')

    # ----------------------------------------------------------
    # Figura 3: perdida total y 3 componentes
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(hist_iter, hist_loss, 'C4-', lw=1.8, label=r'$\mathcal{L}$ total')
    ax.semilogy(hist_iter, hist_L_r,  'C0--', lw=1.2, label=r'$\mathcal{L}_r$ (residuo)')
    ax.semilogy(hist_iter, hist_L_ic, 'C1:', lw=1.2, label=r'$\mathcal{L}_{IC}$')
    ax.semilogy(hist_iter, hist_L_bc, 'C3-.', lw=1.2, label=r'$\mathcal{L}_{BC}$')
    ax.axvline(iter_change, color='gray', linestyle=':',
               label=f'Inicio L-BFGS (iter {iter_change})')
    ax.set_xlabel('Iteracion')
    ax.set_ylabel(r'$\mathcal{L}(\theta)$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'heat_loss.pdf'))
    plt.close(fig)
    print('Generado: heat_loss.pdf')

    # ----------------------------------------------------------
    # Figura 4: decaimiento de la amplitud maxima
    # ----------------------------------------------------------
    amp_pinn = np.max(np.abs(U_pinn), axis=0)
    amp_ref = np.exp(-alpha * np.pi**2 * t_eval)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A: escala lineal
    axes[0].plot(t_eval, amp_ref, 'k-', lw=2, label=r'Analitico: $e^{-\alpha\pi^2 t}$')
    axes[0].plot(t_eval, amp_pinn, 'r--', lw=1.5, label=r'PINN: $\max_x |u_\theta(x,t)|$')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'Amplitud maxima')
    axes[0].set_title('Decaimiento (escala lineal)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Panel B: escala log
    axes[1].semilogy(t_eval, amp_ref, 'k-', lw=2, label=r'Analitico: $e^{-\alpha\pi^2 t}$')
    axes[1].semilogy(t_eval, amp_pinn, 'r--', lw=1.5, label=r'PINN: $\max_x |u_\theta(x,t)|$')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'Amplitud maxima (log)')
    axes[1].set_title('Decaimiento (escala semilog)')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'heat_decay.pdf'))
    plt.close(fig)
    print('Generado: heat_decay.pdf')


if __name__ == '__main__':
    main()