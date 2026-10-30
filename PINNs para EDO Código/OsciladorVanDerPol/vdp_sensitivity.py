import os
import json
import time
import argparse
import numpy as np
import torch

from vdp_pinn import (
    VdPModel, train, evaluate, relative_l2, reference_solution,
    print_device_info,
    MU_BASE, T_FINAL, Y0, N_FREQS_DEFAULT, LAMBDA_IC_DEFAULT,
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(config, n_adam, n_lbfgs, n_coll=2000):
    """
    Ejecuta una sola configuracion y devuelve (e_y1, e_y2, time, loss_final).
    """
    mu        = config.get('mu',          MU_BASE)
    n_layers  = config.get('n_layers',    5)
    n_neurons = config.get('n_neurons',   20)
    activation= config.get('activation',  'tanh')
    n_freqs   = config.get('n_freqs',     N_FREQS_DEFAULT)
    lambda_ic = config.get('lambda_ic',   LAMBDA_IC_DEFAULT)

    # Referencia Radau para este mu
    t_ref, y1_ref, y2_ref = reference_solution(mu, T=T_FINAL)

    # Modelo (semilla fija para reproducibilidad)
    torch.manual_seed(42)
    np.random.seed(42)
    model = VdPModel(
        n_layers=n_layers, n_neurons=n_neurons, activation=activation,
        n_freqs=n_freqs, T=T_FINAL,
    )

    # Entrenamiento
    t0 = time.time()
    history, L_final = train(
        model, mu=mu, T=T_FINAL,
        n_adam=n_adam, n_lbfgs=n_lbfgs, n_coll=n_coll,
        lambda_ic=lambda_ic, verbose=False, log_every=20000,
    )
    elapsed = time.time() - t0

    # Evaluacion
    y1_pinn, y2_pinn = evaluate(model, t_ref)
    e_y1 = relative_l2(y1_pinn, y1_ref)
    e_y2 = relative_l2(y2_pinn, y2_ref)

    return {
        'e_y1': e_y1,
        'e_y2': e_y2,
        'time': elapsed,
        'loss_final': L_final,
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
        print(f"    -> E_L2(y1)={r['e_y1']:.3e}  E_L2(y2)={r['e_y2']:.3e}  "
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
        print(f"    -> E_L2(y1)={r['e_y1']:.3e}  E_L2(y2)={r['e_y2']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def study_lambda_ic(n_adam, n_lbfgs):
    print('=' * 65)
    print(' ESTUDIO 3: PONDERACION DE LA CONDICION INICIAL')
    print('=' * 65)
    configs = [
        {'lambda_ic': 1.0,    'label': 'lambda_ic = 1'},
        {'lambda_ic': 100.0,  'label': 'lambda_ic = 100 (base)'},
        {'lambda_ic': 1000.0, 'label': 'lambda_ic = 1000'},
    ]
    results = []
    for c in configs:
        print(f"  Config: {c['label']:30s}", flush=True)
        r = run_single(c, n_adam, n_lbfgs)
        print(f"    -> E_L2(y1)={r['e_y1']:.3e}  E_L2(y2)={r['e_y2']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def study_mu(n_adam, n_lbfgs):
    print('=' * 65)
    print(' ESTUDIO 4: PARAMETRO DE NO LINEALIDAD MU (RIGIDEZ)')
    print('=' * 65)
    configs = [
        {'mu': 1.0, 'label': 'mu = 1 (base)'},
        {'mu': 2.0, 'label': 'mu = 2'},
        {'mu': 3.0, 'label': 'mu = 3'},
        {'mu': 5.0, 'label': 'mu = 5'},
    ]
    results = []
    for c in configs:
        print(f"  Config: {c['label']:30s}", flush=True)
        r = run_single(c, n_adam, n_lbfgs)
        print(f"    -> E_L2(y1)={r['e_y1']:.3e}  E_L2(y2)={r['e_y2']:.3e}  "
              f"t={r['time']:.1f}s  L={r['loss_final']:.3e}")
        results.append({**c, **r})
    return results


def main():
    parser = argparse.ArgumentParser(description='Estudios de sensibilidad PINN VdP')
    parser.add_argument('--quick', action='store_true',
                        help='Modo rapido: 1500 Adam + 400 LBFGS (verificacion)')
    parser.add_argument('--studies', nargs='+',
                        default=['topology', 'activation', 'lambda_ic', 'mu'],
                        choices=['topology', 'activation', 'lambda_ic', 'mu'],
                        help='Estudios a ejecutar')
    args = parser.parse_args()

    if args.quick:
        n_adam, n_lbfgs = 1500, 400
        print('=' * 65)
        print(' MODO RAPIDO (verificacion): 1500 Adam + 400 L-BFGS')
        print('=' * 65)
    else:
        n_adam, n_lbfgs = 8000, 2000
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
    if 'lambda_ic' in args.studies:
        all_results['lambda_ic'] = study_lambda_ic(n_adam, n_lbfgs)
        print()
    if 'mu' in args.studies:
        all_results['mu'] = study_mu(n_adam, n_lbfgs)
        print()

    # Guardado
    out_file = os.path.join(OUTPUT_DIR, 'sensitivity.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('=' * 65)
    print(f" RESUMEN GLOBAL  (tiempo total: {time.time()-t_global:.1f} s)")
    print('=' * 65)
    for study_name in ['topology', 'activation', 'lambda_ic', 'mu']:
        if study_name not in all_results:
            continue
        print(f"\n  {study_name.upper()}:")
        for r in all_results[study_name]:
            print(f"    {r['label']:30s}  "
                  f"E(y1)={r['e_y1']:.3e}  E(y2)={r['e_y2']:.3e}")
    print(f"\n  Resultados guardados en: {out_file}\n")


if __name__ == '__main__':
    main()