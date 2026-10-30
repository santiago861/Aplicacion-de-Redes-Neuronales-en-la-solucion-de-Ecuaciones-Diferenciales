import os
import json
import time
import numpy as np
import torch

from poisson_pinn import (
    PoissonModel, train, evaluate, evaluate_residual,
    relative_l2, reference_solution,
    print_device_info, DEVICE,
    L_BASE, K_MODES_BASE, N_EVAL,
    N_FREQS_X_DEFAULT, LAMBDA_BC_DEFAULT,
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_base_experiment():
    print("=" * 65)
    print(" EXPERIMENTO BASE Ecuacion de Poisson 1D (PINN)")
    print(f" k_modes={K_MODES_BASE}, L={L_BASE}, f(x)=pi^2 sin(pi x)")
    print(f" 5 capas x 20 neur, tanh, K_x={N_FREQS_X_DEFAULT}")
    print_device_info()
    print("=" * 65)

    # --- Modelo ---
    torch.manual_seed(42)
    np.random.seed(42)
    model = PoissonModel(
        n_layers=5, n_neurons=20, activation='tanh',
        n_freqs=N_FREQS_X_DEFAULT, L=L_BASE,
    )

    # --- Entrenamiento ---
    t0 = time.time()
    history, L_final = train(
        model, k_modes=K_MODES_BASE, L=L_BASE,
        n_adam=5000, n_lbfgs=2500, n_interior=2000,
        lambda_r=1.0, lambda_bc=LAMBDA_BC_DEFAULT,
        log_every=200,
    )
    elapsed = time.time() - t0

    iter_change = history.get('iter_phase_change', 5000)

    t_adam = history.get('time_adam', elapsed * iter_change / max(history['iter'][-1], iter_change + 1))
    t_lbfgs = history.get('time_lbfgs', elapsed - t_adam)

    # --- Evaluacion ---
    x_eval = np.linspace(0.0, L_BASE, N_EVAL)
    u_pinn = evaluate(model, x_eval)
    u_ref = reference_solution(x_eval, k_modes=K_MODES_BASE)
    r_pinn = evaluate_residual(model, x_eval, k_modes=K_MODES_BASE)

    e_l2 = relative_l2(u_pinn, u_ref)
    e_max = float(np.max(np.abs(u_pinn - u_ref)))
    r_max = float(np.max(np.abs(r_pinn)))

    print(f"\n  Tiempo total:        {elapsed:.1f} s")
    print(f"  Tiempo Adam (est):   {t_adam:.1f} s")
    print(f"  Tiempo L-BFGS (est): {t_lbfgs:.1f} s")
    print(f"  Perdida final:       {L_final:.4e}")
    print(f"  Error L^2 relativo:  {e_l2:.4e}")
    print(f"  Error maximo:        {e_max:.4e}")
    print(f"  Residuo maximo:      {r_max:.4e}")

    L_r_f  = history['L_r'][-1]
    L_bc_f = history['L_bc'][-1]

    # --- Guardado ---
    np.savez(os.path.join(OUTPUT_DIR, 'base_poisson.npz'),
             x_eval=x_eval, u_pinn=u_pinn, u_ref=u_ref, r_pinn=r_pinn,
             hist_iter=np.array(history['iter']),
             hist_loss=np.array(history['loss']),
             hist_L_r=np.array(history['L_r']),
             hist_L_bc=np.array(history['L_bc']),
             iter_phase_change=iter_change,
             k_modes=K_MODES_BASE)

    with open(os.path.join(OUTPUT_DIR, 'base_poisson.json'), 'w') as f:
        json.dump({
            'config': {
                'k_modes': K_MODES_BASE, 'L': L_BASE,
                'n_layers': 5, 'n_neurons': 20,
                'activation': 'tanh',
                'n_freqs': N_FREQS_X_DEFAULT,
                'lambda_r': 1.0, 'lambda_bc': LAMBDA_BC_DEFAULT,
                'n_adam': 5000, 'n_lbfgs': 2500,
                'n_interior': 2000,
                'device': str(DEVICE),
            },
            'results': {
                'time_total': elapsed,
                'time_adam':  t_adam,
                'time_lbfgs': t_lbfgs,
                'loss_final': L_final,
                'L_r_final':  L_r_f,
                'L_bc_final': L_bc_f,
                'e_l2':  e_l2,
                'e_max': e_max,
                'r_max': r_max,
            }
        }, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'base_poisson.pt'))
    print(f"\n  Resultados guardados en  {OUTPUT_DIR}/")


if __name__ == '__main__':
    run_base_experiment()