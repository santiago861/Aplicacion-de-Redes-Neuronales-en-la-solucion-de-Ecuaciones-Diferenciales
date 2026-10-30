import os
import json
import time
import argparse
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


def run_single(config, n_adam, n_lbfgs, n_interior=2000, n_ic=200, n_bc=200):
    """
    Ejecuta una sola configuracion y devuelve los indicadores de calidad.
    """
    alpha       = config.get('alpha',       ALPHA_BASE)
    n_layers    = config.get('n_layers',    5)
    n_neurons   = config.get('n_neurons',   20)
    activation  = config.get('activation',  'tanh')
    lambda_data = config.get('lambda_data', LAMBDA_DATA_DEFAULT)

    # Modelo
    torch.manual_seed(42)
    np.random.seed(42)
    model = HeatModel(
        n_layers=n_layers, n_neurons=n_neurons, activation=activation,
        n_freqs_x=N_FREQS_X_DEFAULT, n_freqs_t=N_FREQS_T_DEFAULT,
        L=L_BASE, T=T_FINAL,
    )

    # Entrenamiento
    t0 = time.time()
    history, L_final = train(
        model, alpha=alpha, L=L_BASE, T=T_FINAL,
        n_adam=n_adam, n_lbfgs=n_lbfgs,
        n_interior=n_interior, n_ic=n_ic, n_bc=n_bc,
        lr_adam=1e-3, lambda_r=1.0, lambda_data=lambda_data,
        verbose=False, log_every=100000,
    )
    elapsed = time.time() - t0

    # Evaluacion sobre malla densa
    x_eval = np.linspace(0.0, L_BASE, N_EVAL_X)
    t_eval = np.linspace(0.0, T_FINAL, N_EVAL_T)
    X_grid, T_grid, U_pinn = evaluate_grid(model, x_eval, t_eval)
    U_ref = reference_solution(X_grid, T_grid, alpha=alpha)

    e_l2 = relative_l2(U_pinn, U_ref)
    e_max = float(np.max(np.abs(U_pinn - U_ref)))

    return {
        'e_l2':       e_l2,
        'e_max':      e_max,
        'loss_final': L_final,
        'time':       elapsed,
    }


def study_topology(n_adam, n_lbfgs):
    print('=' * 65)
    print(' ESTUDIO 1: TOPOLOGIA DE LA RED')
    print('=' * 65)
    configs = [
        {'n_layers': 3, 'n_neurons': 20, 'label': '3 capas x 20 neur'},
        {'n_layers': 5, 'n_neurons': 20, 'label': '5 capas x 20 neur (base)'},
        {'n_layers': 5, 'n_neurons': 50, 'label': '5 capas x 50 neur'},
        {'n_layers': 7, 'n_neurons': 80, 'label': '7 capas x 80 neur'},
    ]
    results = []
    for c in configs:
        print(f"  Config: {c['label']:30s}", flush=True)
        r = run_single(c, n_adam, n_lbfgs)
        print(f"    -> E_L2={r['e_l2']:.3e}  E_max={r['e_max']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def study_activation(n_adam, n_lbfgs):
    print('=' * 65)
    print(' ESTUDIO 2: FUNCION DE ACTIVACION')
    print('=' * 65)
    configs = [
        {'activation': 'tanh',    'label': 'tanh (base)'},
        {'activation': 'sigmoid', 'label': 'sigmoid'},
        {'activation': 'swish',   'label': 'swish'},
    ]
    results = []
    for c in configs:
        print(f"  Config: {c['label']:30s}", flush=True)
        r = run_single(c, n_adam, n_lbfgs)
        print(f"    -> E_L2={r['e_l2']:.3e}  E_max={r['e_max']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def study_lambda(n_adam, n_lbfgs):
    print('=' * 65)
    print(' ESTUDIO 3: PONDERACION DE TERMINOS DE DATOS')
    print('=' * 65)
    configs = [
        {'lambda_data': 10.0,   'label': 'lambda_data = 10'},
        {'lambda_data': 100.0,  'label': 'lambda_data = 100 (base)'},
        {'lambda_data': 1000.0, 'label': 'lambda_data = 1000'},
    ]
    results = []
    for c in configs:
        print(f"  Config: {c['label']:30s}", flush=True)
        r = run_single(c, n_adam, n_lbfgs)
        print(f"    -> E_L2={r['e_l2']:.3e}  E_max={r['e_max']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def study_alpha(n_adam, n_lbfgs):
    print('=' * 65)
    print(' ESTUDIO 4: DIFUSIVIDAD TERMICA ALPHA')
    print('=' * 65)
    configs = [
        {'alpha': 0.05, 'label': 'alpha = 0.05'},
        {'alpha': 0.1,  'label': 'alpha = 0.1 (base)'},
        {'alpha': 0.5,  'label': 'alpha = 0.5'},
        {'alpha': 1.0,  'label': 'alpha = 1.0'},
    ]
    results = []
    for c in configs:
        print(f"  Config: {c['label']:30s}", flush=True)
        r = run_single(c, n_adam, n_lbfgs)
        print(f"    -> E_L2={r['e_l2']:.3e}  E_max={r['e_max']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def main():
    parser = argparse.ArgumentParser(description='Estudios de sensibilidad PINN calor 1D')
    parser.add_argument('--quick', action='store_true',
                        help='Modo rapido: 1500 Adam + 500 L-BFGS')
    parser.add_argument('--studies', nargs='+',
                        default=['topology', 'activation', 'lambda', 'alpha'],
                        choices=['topology', 'activation', 'lambda', 'alpha'],
                        help='Estudios a ejecutar')
    args = parser.parse_args()

    if args.quick:
        n_adam, n_lbfgs = 1500, 500
        print('=' * 65)
        print(' MODO RAPIDO (verificacion): 1500 Adam + 500 L-BFGS')
        print('=' * 65)
    else:
        n_adam, n_lbfgs = 5000, 2500
        print('=' * 65)
        print(f' MODO COMPLETO: {n_adam} Adam + {n_lbfgs} L-BFGS')
        print('=' * 65)

    print_device_info()
    print()

    t_global = time.time()
    all_results = {'config': {'n_adam': n_adam, 'n_lbfgs': n_lbfgs}}

    if 'topology' in args.studies:
        all_results['topology'] = study_topology(n_adam, n_lbfgs)
        print()
    if 'activation' in args.studies:
        all_results['activation'] = study_activation(n_adam, n_lbfgs)
        print()
    if 'lambda' in args.studies:
        all_results['lambda'] = study_lambda(n_adam, n_lbfgs)
        print()
    if 'alpha' in args.studies:
        all_results['alpha'] = study_alpha(n_adam, n_lbfgs)
        print()

    out_file = os.path.join(OUTPUT_DIR, 'sensitivity_heat.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('=' * 65)
    print(f" RESUMEN GLOBAL  (tiempo total: {time.time()-t_global:.1f} s)")
    print('=' * 65)
    for study_name in ['topology', 'activation', 'lambda', 'alpha']:
        if study_name not in all_results:
            continue
        print(f"\n  {study_name.upper()}:")
        for r in all_results[study_name]:
            print(f"    {r['label']:30s}  "
                  f"E_L2={r['e_l2']:.3e}  E_max={r['e_max']:.3e}")
    print(f"\n  Resultados guardados en: {out_file}\n")


if __name__ == '__main__':
    main()