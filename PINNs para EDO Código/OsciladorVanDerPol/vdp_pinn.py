import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


DEVICE = get_device()
DTYPE = torch.float32


# -------------------------------------------------------------
# Parametros del problema
# -------------------------------------------------------------
MU_BASE = 1.0
T_FINAL = 10.0
Y0 = [2.0, 0.0]
N_EVAL = 2001
N_FREQS_DEFAULT = 5
LAMBDA_IC_DEFAULT = 100.0


# -------------------------------------------------------------
# Solucion de referencia con integrador implicito Radau
# -------------------------------------------------------------
def vdp_rhs(t, y, mu):
    return [y[1], mu * (1.0 - y[0]**2) * y[1] - y[0]]


def reference_solution(mu=MU_BASE, T=T_FINAL, y0=None, n_eval=N_EVAL):
    if y0 is None:
        y0 = Y0
    t_eval = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(
        vdp_rhs, [0.0, T], y0, args=(mu,),
        method='Radau', t_eval=t_eval,
        rtol=1e-10, atol=1e-10, max_step=0.05,
    )
    assert sol.success, f"Radau fallo: {sol.message}"
    return t_eval, sol.y[0], sol.y[1]


# -------------------------------------------------------------
# Funciones de activacion
# -------------------------------------------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ACTIVATIONS = {
    'tanh':    nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'swish':   Swish,
}


# -------------------------------------------------------------
# Embedding de Fourier
# -------------------------------------------------------------
class FourierEmbedding(nn.Module):
    """phi(t) = [t/T, sin(omega_k t), cos(omega_k t)], omega_k = 2*pi*k/T."""

    def __init__(self, n_freqs=N_FREQS_DEFAULT, T=T_FINAL):
        super().__init__()
        freqs = torch.arange(1, n_freqs + 1, dtype=DTYPE)
        self.register_buffer('omega', 2.0 * np.pi * freqs / T)
        self.T = T
        self.n_freqs = n_freqs

    @property
    def output_dim(self):
        return 1 + 2 * self.n_freqs

    def forward(self, t):
        x = t * self.omega.view(1, -1)
        return torch.cat([t / self.T, torch.sin(x), torch.cos(x)], dim=1)


# -------------------------------------------------------------
# Modelo: MLP simple con embedding de Fourier modesto.
# Condicion inicial suave
# -------------------------------------------------------------
class VdPModel(nn.Module):
    """
    MLP: phi(t) -> 5 capas x 20 neuronas (tanh) -> (y1, y2).
    """

    def __init__(self, n_layers=5, n_neurons=20, activation='tanh',
                 n_freqs=N_FREQS_DEFAULT, T=T_FINAL):
        super().__init__()
        Act = ACTIVATIONS[activation]
        self.embed = FourierEmbedding(n_freqs=n_freqs, T=T)
        in_dim = self.embed.output_dim

        layers = [nn.Linear(in_dim, n_neurons), Act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), Act()]
        layers += [nn.Linear(n_neurons, 2)]
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.T = T

    def forward(self, t):
        return self.net(self.embed(t))


# -------------------------------------------------------------
# Perdida: lambda_ic * L_IC + L_residual
# -------------------------------------------------------------
def pinn_loss(model, t_coll, mu, y0, lambda_ic=LAMBDA_IC_DEFAULT, lambda_res=1.0):
    """
    L = lambda_ic * || u(0) - y_0 ||^2 + lambda_res * (1/N) sum_i (f1^2 + f2^2)
    """
    # Residuo
    t = t_coll.clone().requires_grad_(True)
    u = model(t)
    y1, y2 = u[:, 0:1], u[:, 1:2]
    dy1_dt = torch.autograd.grad(y1, t, torch.ones_like(y1), create_graph=True)[0]
    dy2_dt = torch.autograd.grad(y2, t, torch.ones_like(y2), create_graph=True)[0]
    f1 = dy1_dt - y2
    f2 = dy2_dt - mu * (1.0 - y1**2) * y2 + y1
    L_res = torch.mean(f1**2 + f2**2)

    # Condicion inicial (suave)
    t0 = torch.zeros(1, 1, dtype=DTYPE).to(t_coll.device)
    u0 = model(t0)
    L_ic = (u0[0, 0] - y0[0])**2 + (u0[0, 1] - y0[1])**2

    return lambda_ic * L_ic + lambda_res * L_res, L_ic, L_res


# -------------------------------------------------------------
# Bucle de entrenamiento Adam -> L-BFGS
# -------------------------------------------------------------
def train(model, mu=MU_BASE, T=T_FINAL,
          n_adam=8000, n_lbfgs=2000, n_coll=2000,
          lr_adam=1e-3, lambda_ic=LAMBDA_IC_DEFAULT, seed=42,
          verbose=True, log_every=200, device=None):
    if device is None:
        device = DEVICE

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_coll = (torch.rand(n_coll, 1, dtype=DTYPE) * T).to(device)
    model = model.to(device)

    history = {'iter': [], 'loss_total': [], 'L_ic': [], 'L_res': []}
    g_iter = [0]

    def record(L, L_ic, L_res):
        if g_iter[0] % log_every == 0:
            history['iter'].append(g_iter[0])
            history['loss_total'].append(L.item())
            history['L_ic'].append(L_ic.item())
            history['L_res'].append(L_res.item())
        g_iter[0] += 1

    # Fase 1: Adam
    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
    for i in range(n_adam):
        opt.zero_grad()
        L, L_ic, L_res = pinn_loss(model, t_coll, mu, Y0, lambda_ic)
        L.backward()
        opt.step()
        record(L, L_ic, L_res)
        if verbose and i % 1000 == 0:
            print(f"  Adam {i:6d}: L={L.item():.4e}  L_ic={L_ic.item():.4e}  L_res={L_res.item():.4e}")

    # Fase 2: L-BFGS
    iter_at_phase_change = g_iter[0]
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=n_lbfgs,
        tolerance_grad=1e-11, tolerance_change=1e-13,
        history_size=100, line_search_fn='strong_wolfe',
    )

    def closure():
        opt.zero_grad()
        L, L_ic, L_res = pinn_loss(model, t_coll, mu, Y0, lambda_ic)
        L.backward()
        record(L, L_ic, L_res)
        return L

    opt.step(closure)
    L, L_ic, L_res = pinn_loss(model, t_coll, mu, Y0, lambda_ic)
    if verbose:
        print(f"  LBFGS fin:  L={L.item():.4e}  L_ic={L_ic.item():.4e}  L_res={L_res.item():.4e}")

    history['iter_phase_change'] = iter_at_phase_change
    return history, L.item()


# -------------------------------------------------------------
# Evaluacion y metricas
# -------------------------------------------------------------
def evaluate(model, t_array, device=None):
    if device is None:
        device = DEVICE
    model.eval()
    with torch.no_grad():
        t = torch.tensor(t_array.reshape(-1, 1), dtype=DTYPE).to(device)
        u = model(t).cpu().numpy()
    return u[:, 0], u[:, 1]


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def print_device_info():
    print(f"  Dispositivo: {DEVICE}")
    if DEVICE.type == 'mps':
        print(f"  Backend MPS (Apple Silicon) activado")
    elif DEVICE.type == 'cuda':
        print(f"  Backend CUDA - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Backend CPU - sin aceleracion grafica")