"""
neural_network.py
─────────────────
Car brain: feedforward network evolved via genetic operators.

Architecture
  8 inputs  →  [W1, b1]  →  16 hidden (tanh)  →  [W2, b2]  →  2 outputs (sigmoid)
  Input  0-6 : distance sensors (normalised 0-1)
  Input  7   : speed (normalised 0-1)
  Output 0   : steer  → remapped to [-1, +1]
  Output 1   : throttle → kept as [0, 1]

Genetic operators
  mutate()    Gaussian perturbation with per-layer adaptive scale
  crossover() Uniform crossover (gene-level mask)
  copy_from() Deep copy of all weight arrays
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  ARCHITECTURE CONSTANTS
# ─────────────────────────────────────────────────────────────
_N_IN   = 8
_N_HID  = 16   # increased from 14 — more expressive hidden layer
_N_OUT  = 2
_CLIP   = 500.0


# ─────────────────────────────────────────────────────────────
#  ACTIVATIONS
# ─────────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_CLIP, _CLIP)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -_CLIP, _CLIP))


# ─────────────────────────────────────────────────────────────
#  CAR BRAIN
# ─────────────────────────────────────────────────────────────
class CarBrain:
    """
    Feedforward neural network for a single car.

    Weights are initialised with Xavier / Glorot uniform scaling so that
    activations start in a healthy range — much better than plain randn * 0.5
    for sigmoid/tanh networks.

    Parameters are stored as flat numpy arrays for fast copying and mutation.
    """

    __slots__ = ('W1', 'b1', 'W2', 'b2', '_fitness_cache')

    def __init__(self, zero_init: bool = False):
        if zero_init:
            self.W1 = np.zeros((_N_IN,  _N_HID))
            self.b1 = np.zeros(_N_HID)
            self.W2 = np.zeros((_N_HID, _N_OUT))
            self.b2 = np.zeros(_N_OUT)
        else:
            # Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
            lim1 = np.sqrt(6.0 / (_N_IN  + _N_HID))
            lim2 = np.sqrt(6.0 / (_N_HID + _N_OUT))
            self.W1 = np.random.uniform(-lim1, lim1, (_N_IN,  _N_HID))
            self.b1 = np.zeros(_N_HID)
            self.W2 = np.random.uniform(-lim2, lim2, (_N_HID, _N_OUT))
            self.b2 = np.zeros(_N_OUT)

        self._fitness_cache: Optional[float] = None   # used by selector

    # ─────────────────────────────────────────────────────────
    #  FORWARD PASS  (single car)
    # ─────────────────────────────────────────────────────────
    def forward(self, sensors: np.ndarray) -> np.ndarray:
        """
        Run one forward pass.

        Parameters
        ----------
        sensors : (8,) float array — normalised inputs

        Returns
        -------
        action : (2,) float array
            action[0]  steer    ∈ [-1, +1]   (left … right)
            action[1]  throttle ∈ [ 0,  1]   (brake … full gas)
        """
        h  = _tanh(sensors @ self.W1 + self.b1)          # hidden: tanh
        a2 = _sigmoid(h      @ self.W2 + self.b2)         # output: sigmoid
        return np.array([(a2[0] - 0.5) * 2.0, a2[1]],
                        dtype=np.float64)

    # Legacy name
    def decidir_acao(self, sensors: np.ndarray) -> np.ndarray:
        return self.forward(sensors)

    # ─────────────────────────────────────────────────────────
    #  BATCHED FORWARD PASS  (all alive cars at once)
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def forward_batch(brains: list, sensor_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorised forward pass for N cars simultaneously.

        Parameters
        ----------
        brains        : list of CarBrain, length N
        sensor_matrix : (N, 8) float array

        Returns
        -------
        actions : (N, 2) float array
        """
        N = len(brains)
        W1 = np.empty((_N_IN,  _N_HID, N), dtype=np.float64)
        b1 = np.empty((_N_HID, N),          dtype=np.float64)
        W2 = np.empty((_N_HID, _N_OUT, N),  dtype=np.float64)
        b2 = np.empty((_N_OUT, N),           dtype=np.float64)

        for k, br in enumerate(brains):
            W1[:, :, k] = br.W1
            b1[:,    k] = br.b1
            W2[:, :, k] = br.W2
            b2[:,    k] = br.b2

        # (N, HID) — tanh hidden layer
        pre_h = (np.einsum('ni,ijn->nj', sensor_matrix, W1)    # (N, HID)
                 + b1.T)                                         # broadcast
        h = _tanh(pre_h)

        # (N, OUT) — sigmoid output layer
        pre_o = (np.einsum('nj,jkn->nk', h, W2)
                 + b2.T)
        a2 = _sigmoid(pre_o)

        actions = np.empty((N, 2), dtype=np.float64)
        actions[:, 0] = (a2[:, 0] - 0.5) * 2.0   # steer:    −1 … +1
        actions[:, 1] =  a2[:, 1]                  # throttle:  0 …  1
        return actions

    # ─────────────────────────────────────────────────────────
    #  GENETIC OPERATORS
    # ─────────────────────────────────────────────────────────
    def mutate(self, rate: float = 0.10, strength: Optional[float] = None):
        """
        Gaussian mutation.

        Each weight is perturbed independently with probability *rate*.
        The perturbation magnitude is drawn from N(0, σ) where σ is:
          • *strength* if provided explicitly, otherwise
          • rate itself (so high mutation rate → larger jumps, matching
            the adaptive schedule in SimulatorBase).

        The mutation is applied layer-by-layer with a slight bias:
          output layer weights use half the strength of hidden weights —
          fine-tuning steering/throttle is cheaper than restructuring
          the hidden representation.
        """
        σ = strength if strength is not None else rate
        for attr, scale in (('W1', 1.0), ('b1', 1.0),
                             ('W2', 0.5), ('b2', 0.5)):
            mat  = getattr(self, attr)
            mask = np.random.rand(*mat.shape) < rate
            if mask.any():
                noise = np.random.randn(*mat.shape) * (σ * scale)
                mat   = mat + np.where(mask, noise, 0.0)
                setattr(self, attr, mat)
        self._fitness_cache = None

    def crossover(self, other: 'CarBrain', blend_alpha: float = 0.0):
        """
        Uniform crossover with optional BLX-α blending.

        *blend_alpha* = 0  (default) → standard gene-level uniform crossover:
            each weight is taken from self or other with equal probability.

        *blend_alpha* > 0 → BLX-α: crossed genes are sampled from
            [min − α·Δ, max + α·Δ] — introduces local diversity without
            a full random re-initialisation.  A value of 0.1–0.3 works well
            in practice.
        """
        for attr in ('W1', 'b1', 'W2', 'b2'):
            a = getattr(self, attr)
            b = getattr(other, attr)
            mask = np.random.rand(*a.shape) > 0.5
            if blend_alpha > 0.0:
                lo    = np.minimum(a, b)
                hi    = np.maximum(a, b)
                delta = hi - lo
                blended = np.random.uniform(lo - blend_alpha * delta,
                                            hi + blend_alpha * delta)
                setattr(self, attr, np.where(mask, a, blended))
            else:
                setattr(self, attr, np.where(mask, a, b))
        self._fitness_cache = None

    def copy_from(self, other: 'CarBrain'):
        """Deep-copy all weight arrays from *other* into self."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self._fitness_cache = None

    def clone(self) -> 'CarBrain':
        """Return a new CarBrain that is an exact copy of self."""
        c = CarBrain(zero_init=True)
        c.copy_from(self)
        return c

    # ─────────────────────────────────────────────────────────
    #  SERIALISATION HELPERS
    # ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Serialise weights to plain Python lists (JSON-ready)."""
        return {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
            'architecture': {
                'n_in': _N_IN, 'n_hid': _N_HID, 'n_out': _N_OUT,
                'hidden_activation': 'tanh',
                'output_activation': 'sigmoid',
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CarBrain':
        """Restore a CarBrain from a dict produced by to_dict()."""
        brain    = cls(zero_init=True)
        brain.W1 = np.array(d['W1'], dtype=np.float64)
        brain.b1 = np.array(d['b1'], dtype=np.float64)
        brain.W2 = np.array(d['W2'], dtype=np.float64)
        brain.b2 = np.array(d['b2'], dtype=np.float64)
        return brain

    # ─────────────────────────────────────────────────────────
    #  DIAGNOSTICS
    # ─────────────────────────────────────────────────────────
    def weight_stats(self) -> dict:
        """Return mean abs-weight and max abs-weight per layer — useful for debugging."""
        out = {}
        for name, mat in (('W1', self.W1), ('b1', self.b1),
                           ('W2', self.W2), ('b2', self.b2)):
            flat = np.abs(mat)
            out[name] = {'mean': float(flat.mean()), 'max': float(flat.max())}
        return out

    def __repr__(self) -> str:
        s = self.weight_stats()
        return (f'<CarBrain  {_N_IN}→{_N_HID}→{_N_OUT}'
                f'  |W1|={s["W1"]["mean"]:.3f}'
                f'  |W2|={s["W2"]["mean"]:.3f}>')


# ─────────────────────────────────────────────────────────────
#  LEGACY ALIAS
# ─────────────────────────────────────────────────────────────
RedeNeuralCarrinho = CarBrain