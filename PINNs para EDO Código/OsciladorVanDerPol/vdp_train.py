import os
import json
import time
import numpy as np
import torch

from vdp_pinn import (
    VdPModel, train, evaluate, relative_l2, reference_solution,
    print_device_info, DEVICE,
    MU_BASE, T_FINAL, Y0, N_FREQS_DEFAULT, LAMBDA_IC_DEFAULT,
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_base_experiment():
    print("=" * 65)
    print(" EXPERIMENTO BASE Van der Pol PINN")
    print(f" mu={MU_BASE}, T={T_FINAL}, 5 capas x 20 neur, tanh, K={N_FREQS_DEFAULT}, CI suave")
    print_device_info()
    print("=" * 65)

    # Solucion de referencia con Radau
    t_ref, y1_ref, y2_ref = reference_solution(MU_BASE, T=T_FINAL)

    # Modelo
    torch.manual_seed(42)
    np.random.seed(42)
    model = VdPModel(
        n_layers=5, n_neurons=20, activation='tanh',
        n_freqs=N_FREQS_DEFAULT, T=T_FINAL,
    )

    # Entrenamiento
    t0 = time.time()
    history, L_final = train(
        model, mu=MU_BASE, T=T_FINAL,
        n_adam=8000, n_lbfgs=2000, n_coll=2000,
        lambda_ic=LAMBDA_IC_DEFAULT,
    )
    elapsed = time.time() - t0

    # Tiempos por fase
    iter_change = history.get('iter_phase_change', 8000)
    adam_frac = iter_change / max(history['iter'][-1], iter_change + 1)
    t_adam = elapsed * adam_frac
    t_lbfgs = elapsed - t_adam

    # Evaluacion
    y1_pinn, y2_pinn = evaluate(model, t_ref)
    e_y1 = relative_l2(y1_pinn, y1_ref)
    e_y2 = relative_l2(y2_pinn, y2_ref)

    print(f"\n  Tiempo total:       {elapsed:.1f} s")
    print(f"  Tiempo Adam (est):  {t_adam:.1f} s")
    print(f"  Tiempo L-BFGS (est):{t_lbfgs:.1f} s")
    print(f"  Perdida final:      {L_final:.4e}")
    print(f"  Error L^2 (y1):     {e_y1:.4e}")
    print(f"  Error L^2 (y2):     {e_y2:.4e}")

    # Guardado
    np.savez(os.path.join(OUTPUT_DIR, 'base.npz'),
             t_ref=t_ref, y1_ref=y1_ref, y2_ref=y2_ref,
             y1_pinn=y1_pinn, y2_pinn=y2_pinn,
             hist_iter=np.array(history['iter']),
             hist_loss=np.array(history['loss_total']),
             hist_L_ic=np.array(history['L_ic']),
             hist_L_res=np.array(history['L_res']),
             iter_phase_change=iter_change)

    with open(os.path.join(OUTPUT_DIR, 'base.json'), 'w') as f:
        json.dump({
            'config': {
                'mu': MU_BASE, 'T': T_FINAL, 'y0': Y0,
                'n_layers': 5, 'n_neurons': 20,
                'activation': 'tanh', 'n_freqs': N_FREQS_DEFAULT,
                'lambda_ic': LAMBDA_IC_DEFAULT,
                'n_adam': 8000, 'n_lbfgs': 2000, 'n_coll': 2000,
                'device': str(DEVICE),
            },
            'results': {
                'time_total': elapsed,
                'time_adam':  t_adam,
                'time_lbfgs': t_lbfgs,
                'loss_final': L_final,
                'e_y1': e_y1,
                'e_y2': e_y2,
            }
        }, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'base.pt'))
    print(f"\n  Resultados guardados en  {OUTPUT_DIR}/")


if __name__ == '__main__':
    run_base_experiment()