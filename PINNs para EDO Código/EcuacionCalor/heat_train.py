import os
import json
import time
import numpy as np
import torch

from heat_pinn import (
    HeatModel, train, evaluate_grid, relative_l2, reference_solution,
    print_device_info, DEVICE,
    ALPHA_BASE, L_BASE, T_FINAL, N_EVAL_X, N_EVAL_T,
    N_FREQS_X_DEFAULT, N_FREQS_T_DEFAULT, LAMBDA_DATA_DEFAULT,
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_base_experiment():
    print("=" * 65)
    print(" EXPERIMENTO BASE Ecuacion del Calor 1D (PINN)")
    print(f" alpha={ALPHA_BASE}, L={L_BASE}, T={T_FINAL}")
    print(f" 5 capas x 20 neur, tanh, K_x={N_FREQS_X_DEFAULT}, K_t={N_FREQS_T_DEFAULT}")
    print_device_info()
    print("=" * 65)

    # --- Modelo ---
    torch.manual_seed(42)
    np.random.seed(42)
    model = HeatModel(
        n_layers=5, n_neurons=20, activation='tanh',
        n_freqs_x=N_FREQS_X_DEFAULT, n_freqs_t=N_FREQS_T_DEFAULT,
        L=L_BASE, T=T_FINAL,
    )

    # --- Entrenamiento ---
    t0 = time.time()
    history, L_final = train(
        model, alpha=ALPHA_BASE, L=L_BASE, T=T_FINAL,
        n_adam=5000, n_lbfgs=2500,
        n_interior=2000, n_ic=200, n_bc=200,
        lambda_r=1.0, lambda_data=LAMBDA_DATA_DEFAULT,
        log_every=200,
    )
    elapsed = time.time() - t0

    # Tiempos estimados por fase
    iter_change = history.get('iter_phase_change', 5000)
    adam_frac = iter_change / max(history['iter'][-1], iter_change + 1)
    t_adam = elapsed * adam_frac
    t_lbfgs = elapsed - t_adam

    # --- Evaluacion sobre malla densa ---
    x_eval = np.linspace(0.0, L_BASE, N_EVAL_X)
    t_eval = np.linspace(0.0, T_FINAL, N_EVAL_T)
    X_grid, T_grid, U_pinn = evaluate_grid(model, x_eval, t_eval)
    U_ref = reference_solution(X_grid, T_grid, alpha=ALPHA_BASE)

    e_l2 = relative_l2(U_pinn, U_ref)
    e_max = float(np.max(np.abs(U_pinn - U_ref)))

    print(f"\n  Tiempo total:        {elapsed:.1f} s")
    print(f"  Tiempo Adam (est):   {t_adam:.1f} s")
    print(f"  Tiempo L-BFGS (est): {t_lbfgs:.1f} s")
    print(f"  Perdida final:       {L_final:.4e}")
    print(f"  Error L^2 relativo:  {e_l2:.4e}")
    print(f"  Error maximo:        {e_max:.4e}")

    # Componentes finales de la perdida
    L_r_f  = history['L_r'][-1]
    L_ic_f = history['L_ic'][-1]
    L_bc_f = history['L_bc'][-1]

    # --- Guardado ---
    np.savez(os.path.join(OUTPUT_DIR, 'base_heat.npz'),
             x_eval=x_eval, t_eval=t_eval,
             X_grid=X_grid, T_grid=T_grid,
             U_pinn=U_pinn, U_ref=U_ref,
             hist_iter=np.array(history['iter']),
             hist_loss=np.array(history['loss']),
             hist_L_r=np.array(history['L_r']),
             hist_L_ic=np.array(history['L_ic']),
             hist_L_bc=np.array(history['L_bc']),
             iter_phase_change=iter_change,
             alpha=ALPHA_BASE)

    with open(os.path.join(OUTPUT_DIR, 'base_heat.json'), 'w') as f:
        json.dump({
            'config': {
                'alpha': ALPHA_BASE, 'L': L_BASE, 'T': T_FINAL,
                'n_layers': 5, 'n_neurons': 20,
                'activation': 'tanh',
                'n_freqs_x': N_FREQS_X_DEFAULT, 'n_freqs_t': N_FREQS_T_DEFAULT,
                'lambda_r': 1.0, 'lambda_data': LAMBDA_DATA_DEFAULT,
                'n_adam': 5000, 'n_lbfgs': 2500,
                'n_interior': 2000, 'n_ic': 200, 'n_bc': 200,
                'device': str(DEVICE),
            },
            'results': {
                'time_total': elapsed,
                'time_adam':  t_adam,
                'time_lbfgs': t_lbfgs,
                'loss_final': L_final,
                'L_r_final':  L_r_f,
                'L_ic_final': L_ic_f,
                'L_bc_final': L_bc_f,
                'e_l2':  e_l2,
                'e_max': e_max,
            }
        }, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'base_heat.pt'))
    print(f"\n  Resultados guardados en  {OUTPUT_DIR}/")


if __name__ == '__main__':
    run_base_experiment()