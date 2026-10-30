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

DATA_FILE = './results/base_poisson.npz'
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontro {DATA_FILE}. Ejecuta poisson_train.py primero.")

    d = np.load(DATA_FILE)
    x = d['x_eval']
    u_pinn = d['u_pinn']
    u_ref = d['u_ref']
    r_pinn = d['r_pinn']
    err = np.abs(u_pinn - u_ref)
    k_modes = int(d['k_modes'])

    hist_iter  = d['hist_iter']
    hist_loss  = d['hist_loss']
    hist_L_r   = d['hist_L_r']
    hist_L_bc  = d['hist_L_bc']
    iter_change = int(d['iter_phase_change'])

    # ----------------------------------------------------------
    # Figura 1: Solucion (PINN vs analitica)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, u_ref,  'k-',  lw=2,   label='Referencia analitica')
    ax.plot(x, u_pinn, 'r--', lw=1.5, label='PINN')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')
    title = (r'Solucion: $u(x) = \sin(\pi x)$'
             if k_modes == 1
             else rf'Solucion: $u(x) = \sum_{{k=1}}^{{{k_modes}}} \sin(k\pi x)$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'poisson_solution.pdf'))
    plt.close(fig)
    print('Generado: poisson_solution.pdf')

    # ----------------------------------------------------------
    # Figura 2: Error puntual (escala semilog)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    eps = max(err.max() * 1e-10, 1e-16)
    ax.semilogy(x, err + eps, 'C3-', lw=1.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|u_\theta(x) - u_{\rm ref}(x)|$')
    ax.set_title('Error puntual (escala semilog)')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'poisson_error.pdf'))
    plt.close(fig)
    print('Generado: poisson_error.pdf')

    # ----------------------------------------------------------
    # Figura 3: Residuo r(x) = -u''(x) - f(x)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, r_pinn, 'C0-', lw=1.5)
    ax.axhline(0, color='k', linestyle=':', lw=0.8)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r"$r_\theta(x) = -u''_\theta(x) - f(x)$")
    ax.set_title('Residuo de la PDE a lo largo del dominio')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'poisson_residual.pdf'))
    plt.close(fig)
    print('Generado: poisson_residual.pdf')

    # ----------------------------------------------------------
    # Figura 4: Perdida total y 2 componentes
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(hist_iter, hist_loss, 'C4-',  lw=1.8, label=r'$\mathcal{L}$ total')
    ax.semilogy(hist_iter, hist_L_r,  'C0--', lw=1.2, label=r'$\mathcal{L}_r$ (residuo)')
    ax.semilogy(hist_iter, hist_L_bc, 'C3-.', lw=1.2, label=r'$\mathcal{L}_{BC}$')
    ax.axvline(iter_change, color='gray', linestyle=':',
               label=f'Inicio L-BFGS (iter {iter_change})')
    ax.set_xlabel('Iteracion')
    ax.set_ylabel(r'$\mathcal{L}(\theta)$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'poisson_loss.pdf'))
    plt.close(fig)
    print('Generado: poisson_loss.pdf')


if __name__ == '__main__':
    main()