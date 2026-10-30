import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import torch.nn as nn
import numpy as np


# -------------------------------------------------------------
# Dispositivo
# -------------------------------------------------------------
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
C_BASE = 2.0
L_BASE = 1.0
T_FINAL = 1.0
N_EVAL_X = 201
N_EVAL_T = 201

N_FREQS_X_DEFAULT = 4
N_FREQS_T_DEFAULT = 4
LAMBDA_DATA_DEFAULT = 100.0


# -------------------------------------------------------------
# Solucion analitica de referencia y condiciones iniciales
# -------------------------------------------------------------
def reference_solution(x, t, c=C_BASE):
    """u(x, t) = sin(pi x) cos(pi c t)."""
    return np.sin(np.pi * x) * np.cos(np.pi * c * t)


def f_ic(x):
    """u(x, 0) = sin(pi x)."""
    return torch.sin(np.pi * x)


def g_ic(x):
    """u_t(x, 0) = 0."""
    return torch.zeros_like(x)


# -------------------------------------------------------------
# Activaciones
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
# Embedding de Fourier 2D
# -------------------------------------------------------------
class FourierEmbedding2D(nn.Module):
    """
    phi(x, t) = [x/L, t/T,
                 sin(omega_x_k x), cos(omega_x_k x),
                 sin(omega_t_k t), cos(omega_t_k t)]
    con omega_x_k = pi k / L  (base sinusoidal compatible con CF Dirichlet)
        omega_t_k = 2 pi k / T
    """
    def __init__(self, n_freqs_x=N_FREQS_X_DEFAULT, n_freqs_t=N_FREQS_T_DEFAULT,
                 L=L_BASE, T=T_FINAL):
        super().__init__()
        freqs_x = torch.arange(1, n_freqs_x + 1, dtype=DTYPE)
        freqs_t = torch.arange(1, n_freqs_t + 1, dtype=DTYPE)
        self.register_buffer('omega_x', np.pi * freqs_x / L)
        self.register_buffer('omega_t', 2.0 * np.pi * freqs_t / T)
        self.L = L
        self.T = T
        self.n_freqs_x = n_freqs_x
        self.n_freqs_t = n_freqs_t

    @property
    def output_dim(self):
        return 2 + 2 * self.n_freqs_x + 2 * self.n_freqs_t

    def forward(self, x, t):
        xx = x * self.omega_x.view(1, -1)
        tt = t * self.omega_t.view(1, -1)
        return torch.cat([
            x / self.L, t / self.T,
            torch.sin(xx), torch.cos(xx),
            torch.sin(tt), torch.cos(tt),
        ], dim=1)


# -------------------------------------------------------------
# Modelo
# -------------------------------------------------------------
class WaveModel(nn.Module):
    """phi(x,t) -> 5 capas x 20 neuronas (tanh) -> u(x,t)."""

    def __init__(self, n_layers=5, n_neurons=20, activation='tanh',
                 n_freqs_x=N_FREQS_X_DEFAULT, n_freqs_t=N_FREQS_T_DEFAULT,
                 L=L_BASE, T=T_FINAL):
        super().__init__()
        Act = ACTIVATIONS[activation]
        self.embed = FourierEmbedding2D(n_freqs_x, n_freqs_t, L, T)
        in_dim = self.embed.output_dim
        layers = [nn.Linear(in_dim, n_neurons), Act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), Act()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.L = L
        self.T = T

    def forward(self, x, t):
        return self.net(self.embed(x, t))


# -------------------------------------------------------------
# Funcion de perdida
# -------------------------------------------------------------
def pinn_loss(model, x_int, t_int, x_ic, x_bc, t_bc, c,
              lambda_r=1.0, lambda_ic1=LAMBDA_DATA_DEFAULT,
              lambda_ic2=LAMBDA_DATA_DEFAULT, lambda_bc=LAMBDA_DATA_DEFAULT):
    """
    L = lambda_r * L_r + lambda_ic1 * L_ic1 + lambda_ic2 * L_ic2 + lambda_bc * L_bc
    """
    # ----- Residuo PDE: u_tt - c^2 u_xx = 0 -----
    x = x_int.clone().requires_grad_(True)
    t = t_int.clone().requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    r = u_tt - c**2 * u_xx
    L_r = torch.mean(r**2)

    # ----- CI 1: u(x, 0) = f(x) -----
    t_zero = torch.zeros_like(x_ic)
    u_ic = model(x_ic, t_zero)
    L_ic1 = torch.mean((u_ic - f_ic(x_ic))**2)

    # ----- CI 2: u_t(x, 0) = g(x) -----
    t_zero_grad = torch.zeros_like(x_ic).requires_grad_(True)
    u_ic_v = model(x_ic, t_zero_grad)
    u_t_ic = torch.autograd.grad(u_ic_v, t_zero_grad,
                                  torch.ones_like(u_ic_v), create_graph=True)[0]
    L_ic2 = torch.mean((u_t_ic - g_ic(x_ic))**2)

    # ----- CF: u(0, t) = u(L, t) = 0 -----
    u_bc = model(x_bc, t_bc)
    L_bc = torch.mean(u_bc**2)

    L_total = (lambda_r * L_r +
               lambda_ic1 * L_ic1 +
               lambda_ic2 * L_ic2 +
               lambda_bc * L_bc)
    comps = {
        'L_r':   L_r.item(),
        'L_ic1': L_ic1.item(),
        'L_ic2': L_ic2.item(),
        'L_bc':  L_bc.item(),
    }
    return L_total, comps


# -------------------------------------------------------------
# Generacion de puntos
# -------------------------------------------------------------
def sample_points(n_interior=5000, n_ic=200, n_bc=200,
                  L=L_BASE, T=T_FINAL, seed=42, device=None):
    if device is None:
        device = DEVICE
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_int = (torch.rand(n_interior, 1, dtype=DTYPE) * L).to(device)
    t_int = (torch.rand(n_interior, 1, dtype=DTYPE) * T).to(device)

    x_ic_pts = (torch.rand(n_ic, 1, dtype=DTYPE) * L).to(device)

    # n_bc puntos en x=0 y otros n_bc en x=L
    t_bc_pts = (torch.rand(2 * n_bc, 1, dtype=DTYPE) * T).to(device)
    x_bc_pts = torch.cat([
        torch.zeros(n_bc, 1, dtype=DTYPE),
        torch.full((n_bc, 1), float(L), dtype=DTYPE),
    ], dim=0).to(device)

    return x_int, t_int, x_ic_pts, x_bc_pts, t_bc_pts


# -------------------------------------------------------------
# Entrenamiento
# -------------------------------------------------------------
def train(model, c=C_BASE, L=L_BASE, T=T_FINAL,
          n_adam=10000, n_lbfgs=2000,
          n_interior=5000, n_ic=200, n_bc=200,
          lr_adam=1e-3, lambda_r=1.0, lambda_data=LAMBDA_DATA_DEFAULT,
          seed=42, verbose=True, log_every=200, device=None):
    if device is None:
        device = DEVICE

    x_int, t_int, x_ic, x_bc, t_bc = sample_points(
        n_interior=n_interior, n_ic=n_ic, n_bc=n_bc,
        L=L, T=T, seed=seed, device=device,
    )
    model = model.to(device)

    history = {'iter': [], 'loss': [],
               'L_r': [], 'L_ic1': [], 'L_ic2': [], 'L_bc': []}
    g_iter = [0]

    def record(L_total, comps):
        if g_iter[0] % log_every == 0:
            history['iter'].append(g_iter[0])
            history['loss'].append(L_total.item())
            history['L_r'].append(comps['L_r'])
            history['L_ic1'].append(comps['L_ic1'])
            history['L_ic2'].append(comps['L_ic2'])
            history['L_bc'].append(comps['L_bc'])
        g_iter[0] += 1

    # --- Adam ---
    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
    for i in range(n_adam):
        opt.zero_grad()
        L_total, comps = pinn_loss(
            model, x_int, t_int, x_ic, x_bc, t_bc, c,
            lambda_r=lambda_r, lambda_ic1=lambda_data,
            lambda_ic2=lambda_data, lambda_bc=lambda_data,
        )
        L_total.backward()
        opt.step()
        record(L_total, comps)
        if verbose and i % 1000 == 0:
            print(f"  Adam {i:6d}: L={L_total.item():.4e}  "
                  f"r={comps['L_r']:.2e}  ic1={comps['L_ic1']:.2e}  "
                  f"ic2={comps['L_ic2']:.2e}  bc={comps['L_bc']:.2e}")

    # --- L-BFGS ---
    iter_at_phase_change = g_iter[0]
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=n_lbfgs,
        tolerance_grad=1e-11, tolerance_change=1e-13,
        history_size=100, line_search_fn='strong_wolfe',
    )

    def closure():
        opt.zero_grad()
        L_total, comps = pinn_loss(
            model, x_int, t_int, x_ic, x_bc, t_bc, c,
            lambda_r=lambda_r, lambda_ic1=lambda_data,
            lambda_ic2=lambda_data, lambda_bc=lambda_data,
        )
        L_total.backward()
        record(L_total, comps)
        return L_total

    opt.step(closure)
    L_total, comps = pinn_loss(
        model, x_int, t_int, x_ic, x_bc, t_bc, c,
        lambda_r=lambda_r, lambda_ic1=lambda_data,
        lambda_ic2=lambda_data, lambda_bc=lambda_data,
    )
    if verbose:
        print(f"  LBFGS fin: L={L_total.item():.4e}  "
              f"r={comps['L_r']:.2e}  ic1={comps['L_ic1']:.2e}  "
              f"ic2={comps['L_ic2']:.2e}  bc={comps['L_bc']:.2e}")

    history['iter_phase_change'] = iter_at_phase_change
    return history, L_total.item()


# -------------------------------------------------------------
# Evaluacion
# -------------------------------------------------------------
def evaluate_grid(model, x_grid, t_grid, device=None):
    """Evalua el modelo sobre la grid (x_grid x t_grid)."""
    if device is None:
        device = DEVICE
    model.eval()
    with torch.no_grad():
        X, T_ = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_flat = torch.tensor(X.flatten().reshape(-1, 1), dtype=DTYPE).to(device)
        t_flat = torch.tensor(T_.flatten().reshape(-1, 1), dtype=DTYPE).to(device)
        u_flat = model(x_flat, t_flat).cpu().numpy().flatten()
        U = u_flat.reshape(X.shape)
    return X, T_, U


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def print_device_info():
    print(f"  Dispositivo: {DEVICE}")
    if DEVICE.type == 'mps':
        print(f"  Backend MPS (Apple Silicon)")
    elif DEVICE.type == 'cuda':
        print(f"  Backend CUDA - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Backend CPU")