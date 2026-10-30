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
L_BASE = 1.0
K_MODES_BASE = 1
N_EVAL = 401
N_FREQS_X_DEFAULT = 4
LAMBDA_BC_DEFAULT = 100.0


# -------------------------------------------------------------
# Termino fuente y solucion analitica
# -------------------------------------------------------------
def source(x, k_modes=K_MODES_BASE):
    """f(x) = sum_{k=1..K} (k pi)^2 sin(k pi x)."""
    f = np.zeros_like(x)
    for k in range(1, k_modes + 1):
        f += (k * np.pi)**2 * np.sin(k * np.pi * x)
    return f


def source_torch(x, k_modes=K_MODES_BASE):
    """Version torch del termino fuente."""
    f = torch.zeros_like(x)
    for k in range(1, k_modes + 1):
        f += (k * np.pi)**2 * torch.sin(k * np.pi * x)
    return f


def reference_solution(x, k_modes=K_MODES_BASE):
    """u(x) = sum_{k=1..K} sin(k pi x)."""
    u = np.zeros_like(x)
    for k in range(1, k_modes + 1):
        u += np.sin(k * np.pi * x)
    return u


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
# Embedding de Fourier 1D
# -------------------------------------------------------------
class FourierEmbedding1D(nn.Module):
    """phi(x) = [x/L, sin(omega_k x), cos(omega_k x)] para k=1..K_x.

    omega_k = pi k / L coincide con los eigenmodos de Dirichlet del
    problema y proporciona una base compatible con la CF homogenea
    """
    def __init__(self, n_freqs=N_FREQS_X_DEFAULT, L=L_BASE):
        super().__init__()
        freqs = torch.arange(1, n_freqs + 1, dtype=DTYPE)
        self.register_buffer('omega', np.pi * freqs / L)
        self.L = L
        self.n_freqs = n_freqs

    @property
    def output_dim(self):
        return 1 + 2 * self.n_freqs

    def forward(self, x):
        xx = x * self.omega.view(1, -1)
        return torch.cat([x / self.L, torch.sin(xx), torch.cos(xx)], dim=1)


# -------------------------------------------------------------
# Modelo
# -------------------------------------------------------------
class PoissonModel(nn.Module):
    """phi(x) -> 5 capas x 20 neuronas (tanh) -> u(x)"""

    def __init__(self, n_layers=5, n_neurons=20, activation='tanh',
                 n_freqs=N_FREQS_X_DEFAULT, L=L_BASE):
        super().__init__()
        Act = ACTIVATIONS[activation]
        self.embed = FourierEmbedding1D(n_freqs, L)
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

    def forward(self, x):
        return self.net(self.embed(x))


# -------------------------------------------------------------
# Funcion de perdida
# -------------------------------------------------------------
def pinn_loss(model, x_int, x_bc, u_bc, k_modes=K_MODES_BASE,
              lambda_r=1.0, lambda_bc=LAMBDA_BC_DEFAULT):
    """
    L = lambda_r * L_r + lambda_bc * L_bc
    Residuo: -u''(x) - f(x) = 0
    """
    # ----- Residuo PDE: -u_xx - f(x) -----
    x = x_int.clone().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = source_torch(x, k_modes)
    r = -u_xx - f
    L_r = torch.mean(r**2)

    # ----- CF: u(0) = u_L, u(L) = u_R -----
    u_pred_bc = model(x_bc)
    L_bc = torch.mean((u_pred_bc - u_bc)**2)

    L_total = lambda_r * L_r + lambda_bc * L_bc
    comps = {
        'L_r':  L_r.item(),
        'L_bc': L_bc.item(),
    }
    return L_total, comps


# -------------------------------------------------------------
# Generacion de puntos
# -------------------------------------------------------------
def sample_points(n_interior=2000, L=L_BASE, seed=42, device=None,
                  u_L=0.0, u_R=0.0):
    """
    Muestrea n_interior puntos interiores en [0, L] uniformemente
    y fija los dos puntos exactos de frontera.
    """
    if device is None:
        device = DEVICE
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_int = (torch.rand(n_interior, 1, dtype=DTYPE) * L).to(device)
    x_bc = torch.tensor([[0.0], [float(L)]], dtype=DTYPE).to(device)
    u_bc = torch.tensor([[u_L], [u_R]], dtype=DTYPE).to(device)
    return x_int, x_bc, u_bc


# -------------------------------------------------------------
# Entrenamiento
# -------------------------------------------------------------
def train(model, k_modes=K_MODES_BASE, L=L_BASE,
          n_adam=5000, n_lbfgs=2500, n_interior=2000,
          lr_adam=1e-3, lambda_r=1.0, lambda_bc=LAMBDA_BC_DEFAULT,
          seed=42, verbose=True, log_every=200, device=None):
    import time as _time
    if device is None:
        device = DEVICE

    x_int, x_bc, u_bc = sample_points(
        n_interior=n_interior, L=L, seed=seed, device=device,
    )
    model = model.to(device)

    history = {'iter': [], 'loss': [], 'L_r': [], 'L_bc': []}
    g_iter = [0]

    def record(L_total, comps, force=False):
        if force or g_iter[0] % log_every == 0:
            history['iter'].append(g_iter[0])
            history['loss'].append(L_total.item())
            history['L_r'].append(comps['L_r'])
            history['L_bc'].append(comps['L_bc'])
        g_iter[0] += 1

    # --- Adam ---
    t_adam_start = _time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
    for i in range(n_adam):
        opt.zero_grad()
        L_total, comps = pinn_loss(
            model, x_int, x_bc, u_bc, k_modes=k_modes,
            lambda_r=lambda_r, lambda_bc=lambda_bc,
        )
        L_total.backward()
        opt.step()
        record(L_total, comps)
        if verbose and i % 1000 == 0:
            print(f"  Adam {i:6d}: L={L_total.item():.4e}  "
                  f"r={comps['L_r']:.2e}  bc={comps['L_bc']:.2e}")
    t_adam = _time.time() - t_adam_start

    # --- L-BFGS ---
    iter_at_phase_change = g_iter[0]
    t_lbfgs_start = _time.time()
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=n_lbfgs,
        tolerance_grad=1e-11, tolerance_change=1e-13,
        history_size=100, line_search_fn='strong_wolfe',
    )

    def closure():
        opt.zero_grad()
        L_total, comps = pinn_loss(
            model, x_int, x_bc, u_bc, k_modes=k_modes,
            lambda_r=lambda_r, lambda_bc=lambda_bc,
        )
        L_total.backward()

        record(L_total, comps, force=True)
        return L_total

    opt.step(closure)
    t_lbfgs = _time.time() - t_lbfgs_start

    L_total, comps = pinn_loss(
        model, x_int, x_bc, u_bc, k_modes=k_modes,
        lambda_r=lambda_r, lambda_bc=lambda_bc,
    )
    
    record(L_total, comps, force=True)
    if verbose:
        print(f"  LBFGS fin: L={L_total.item():.4e}  "
              f"r={comps['L_r']:.2e}  bc={comps['L_bc']:.2e}")

    history['iter_phase_change'] = iter_at_phase_change
    history['time_adam'] = t_adam
    history['time_lbfgs'] = t_lbfgs
    return history, L_total.item()


# -------------------------------------------------------------
# Evaluacion
# -------------------------------------------------------------
def evaluate(model, x_array, device=None):
    """Evalua u_theta(x) sobre x_array"""
    if device is None:
        device = DEVICE
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_array.reshape(-1, 1), dtype=DTYPE).to(device)
        u = model(x).cpu().numpy().flatten()
    return u


def evaluate_residual(model, x_array, k_modes=K_MODES_BASE, device=None):
    """Evalua r(x) = -u''(x) - f(x) sobre x_array."""
    if device is None:
        device = DEVICE
    model.eval()
    x = torch.tensor(x_array.reshape(-1, 1), dtype=DTYPE,
                     requires_grad=True).to(device)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=False)[0]
    f = source_torch(x, k_modes)
    r = (-u_xx - f).detach().cpu().numpy().flatten()
    return r


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