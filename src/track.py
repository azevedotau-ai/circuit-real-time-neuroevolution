
import sys
import numpy as np
import json
import os
import time
import logging
from collections import deque
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  RESOURCE PATH  (dev + PyInstaller)
# ─────────────────────────────────────────────────────────────
def _resource_path(relative: str) -> str:
    base = getattr(sys, '_MEIPASS',
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, relative)


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    path = _resource_path('setting.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


CONFIG = load_config()
_COR   = CONFIG['ui_colors']  


# ─────────────────────────────────────────────────────────────
#  TRACK JSON  (supports both old pt-BR keys and new en keys)
# ─────────────────────────────────────────────────────────────
def _load_track_json() -> dict:
    # Try new name first, fall back to legacy name
    for name in ('track.json', 'pista.json'):
        p = _resource_path(name)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            log.info(f'[track] Loaded "{p}"')
            return d
    raise FileNotFoundError('Neither track.json nor pista.json found.')


def _key(d: dict, *candidates, default=None):
   
    for k in candidates:
        if k in d:
            return d[k]
    return default


_tj = _load_track_json()


_raw_pts      = _key(_tj, 'control_points', 'pontos_controle', default=[])
_CTRL_POINTS  = np.array(_raw_pts, dtype=np.float64)
_NOME_PISTA   = _key(_tj, 'name', 'nome', default='Track')
_TRACK_WIDTH  = float(_key(_tj, 'track_width', 'largura_pista', default=1.8))


CONFIG.setdefault('track', {})['track_width'] = _TRACK_WIDTH
CONFIG.setdefault('pista', {})['largura_pista'] = _TRACK_WIDTH  


_sl = _key(_tj, 'start', 'largada', default={})
CONFIG.setdefault('start_line', {}).update({
    'x':          float(_key(_sl, 'x',              default=0.0)),
    'y':          float(_key(_sl, 'y',              default=0.0)),
    'angle':      float(_key(_sl, 'line_angle_deg', 'angulo_linha_graus', default=0.0)),
    'width':      float(_key(_sl, 'line_width',     'largura_linha',      default=_TRACK_WIDTH)),
    'height':     0.4,
    'num_stripes': 14,
})
CONFIG.setdefault('linha_largada', CONFIG['start_line'])  
CONFIG['cars']['initial_angle'] = float(
    _key(_sl, 'car_angle_deg', 'angulo_carro_graus', default=0.0))
CONFIG['carros'] = CONFIG['cars']  


_fl = _key(_tj, 'finish', 'chegada', default=_sl)
CONFIG.setdefault('finish_line', {}).update({
    'x':      float(_key(_fl, 'x',              default=CONFIG['start_line']['x'])),
    'y':      float(_key(_fl, 'y',              default=CONFIG['start_line']['y'])),
    'angle':  float(_key(_fl, 'line_angle_deg', 'angulo_linha_graus', default=CONFIG['start_line']['angle'])),
    'width':  float(_key(_fl, 'line_width',     'largura_linha',      default=_TRACK_WIDTH)),
    'height': 0.4,
    'num_stripes': 14,
})
CONFIG.setdefault('linha_chegada', CONFIG['finish_line']) 

print(f'[track.json]  "{_NOME_PISTA}"  |  '
      f'{len(_CTRL_POINTS)} control pts  |  width {_TRACK_WIDTH:.3f}')


# ─────────────────────────────────────────────────────────────
#  CATMULL-ROM SPLINE
# ─────────────────────────────────────────────────────────────
def catmull_rom(pts: np.ndarray, n_seg: int = 40) -> np.ndarray:
    """
    Closed Catmull-Rom spline through *pts*.
    Expects pts[0] == pts[-1] (closed loop) or any array of ≥4 points.
    Returns (N * n_seg, 2) array.
    """
    n   = len(pts) - 1   
    ts  = np.linspace(0, 1, n_seg, endpoint=False)
    tt  = ts * ts
    ttt = tt * ts
    out = []
    for i in range(n):
        p0 = pts[i - 1] if i > 0     else pts[n - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 <= n else pts[(i + 2) - n]
        seg = 0.5 * (
            np.outer(-ttt + 2*tt - ts,    p0) +
            np.outer( 3*ttt - 5*tt + 2,   p1) +
            np.outer(-3*ttt + 4*tt + ts,   p2) +
            np.outer( ttt - tt,            p3)
        )
        out.append(seg)
    return np.vstack(out)


# ─────────────────────────────────────────────────────────────
#  PRE-COMPUTED CENTERLINE
# ─────────────────────────────────────────────────────────────
_CENTERLINE: np.ndarray = catmull_rom(_CTRL_POINTS)  # (N, 2)
_CL_X = np.append(_CENTERLINE[:, 0], _CENTERLINE[0, 0])
_CL_Y = np.append(_CENTERLINE[:, 1], _CENTERLINE[0, 1])


_CL_DELTAS    = np.sqrt(np.diff(_CL_X)**2 + np.diff(_CL_Y)**2)
_CL_ARC       = np.concatenate([[0.0], np.cumsum(_CL_DELTAS)])
_CL_TOTAL_LEN = float(_CL_ARC[-1])


_CL_TX = np.diff(_CL_X); _CL_TY = np.diff(_CL_Y)
_CL_TM = np.maximum(np.sqrt(_CL_TX**2 + _CL_TY**2), 1e-12)
_CL_TX /= _CL_TM;  _CL_TY /= _CL_TM


_PAD_X, _PAD_Y = 6.0, 3.0
_TRACK_XLIM = (float(_CL_X.min() - _PAD_X), float(_CL_X.max() + _PAD_X))
_TRACK_YLIM = (float(_CL_Y.min() - _PAD_Y), float(_CL_Y.max() + _PAD_Y))


# ─────────────────────────────────────────────────────────────
#  SDF  (Signed Distance Field — pre-computed for O(1) lookup)
# ─────────────────────────────────────────────────────────────
def _build_sdf(res: float = 0.08) -> Tuple[np.ndarray, float, float, float]:
    """
    Build a 2-D grid where each cell stores squared distance to the
    nearest centerline segment.  Query is O(1) — just an index lookup.
    """
    pad  = 3.0
    xmin = float(_CL_X.min()) - pad;  xmax = float(_CL_X.max()) + pad
    ymin = float(_CL_Y.min()) - pad;  ymax = float(_CL_Y.max()) + pad

    xs = np.arange(xmin, xmax, res, dtype=np.float32)
    ys = np.arange(ymin, ymax, res, dtype=np.float32)
    GX, GY = np.meshgrid(xs, ys)

    ax_ = _CL_X[:-1].astype(np.float32);  ay_ = _CL_Y[:-1].astype(np.float32)
    bx_ = _CL_X[1:].astype(np.float32);   by_ = _CL_Y[1:].astype(np.float32)
    ddx = bx_ - ax_;  ddy = by_ - ay_
    den = np.maximum(ddx*ddx + ddy*ddy, np.float32(1e-12))

    min_d2 = np.full(GX.shape, np.inf, dtype=np.float32)
    for i in range(len(ax_)):
        t  = np.clip(((GX - ax_[i])*ddx[i] + (GY - ay_[i])*ddy[i]) / den[i], 0.0, 1.0)
        d2 = (ax_[i] + t*ddx[i] - GX)**2 + (ay_[i] + t*ddy[i] - GY)**2
        np.minimum(min_d2, d2, out=min_d2)

    return min_d2, xmin, ymin, float(res)


print('[SDF] Pre-computing track distance grid …', end=' ', flush=True)
_t0 = time.perf_counter()
_SDF_GRID, _SDF_XMIN, _SDF_YMIN, _SDF_RES = _build_sdf()
print(f'done  ({_SDF_GRID.shape[1]}×{_SDF_GRID.shape[0]} cells, '
      f'{time.perf_counter() - _t0:.2f}s)')

_HW2 = np.float32((_TRACK_WIDTH / 2.0) ** 2)


# ─────────────────────────────────────────────────────────────
#  COLLISION / MEMBERSHIP QUERIES
# ─────────────────────────────────────────────────────────────
def point_on_track(x: float, y: float) -> bool:
    """O(1) single-point track membership via SDF lookup."""
    xi = int((x - _SDF_XMIN) / _SDF_RES)
    yi = int((y - _SDF_YMIN) / _SDF_RES)
    H, W = _SDF_GRID.shape
    if xi < 0 or yi < 0 or xi >= W or yi >= H:
        return False
    return float(_SDF_GRID[yi, xi]) <= _HW2


ponto_dentro_pista = point_on_track


def points_on_track_batch(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Vectorised track membership for arrays of points.
    Returns bool array of length len(xs).
    """
    xi = ((xs - _SDF_XMIN) / _SDF_RES).astype(np.int32)
    yi = ((ys - _SDF_YMIN) / _SDF_RES).astype(np.int32)
    H, W = _SDF_GRID.shape
    valid = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
    d2    = np.full(len(xs), np.inf, dtype=np.float32)
    d2[valid] = _SDF_GRID[yi[valid], xi[valid]]
    return d2 <= _HW2


_multiponto_dentro_pista = points_on_track_batch


def distance_to_centerline(x: float, y: float) -> float:
    """Euclidean distance from point to nearest centerline segment."""
    xi = int((x - _SDF_XMIN) / _SDF_RES)
    yi = int((y - _SDF_YMIN) / _SDF_RES)
    H, W = _SDF_GRID.shape
    xi = max(0, min(xi, W - 1));  yi = max(0, min(yi, H - 1))
    return float(np.sqrt(_SDF_GRID[yi, xi]))


def track_progress_fraction(x: float, y: float) -> float:
    """
    Returns a value in [0, 1) representing how far along the lap the
    point (x, y) is, measured as arc-length fraction from the start.
    """
    ax_ = _CL_X[:-1];  ay_ = _CL_Y[:-1]
    bx_ = _CL_X[1:];   by_ = _CL_Y[1:]
    dx  = bx_ - ax_;   dy  = by_ - ay_
    den = np.maximum(dx*dx + dy*dy, 1e-12)
    t   = np.clip(((x - ax_)*dx + (y - ay_)*dy) / den, 0.0, 1.0)
    d2  = (ax_ + t*dx - x)**2 + (ay_ + t*dy - y)**2
    idx = int(np.argmin(d2))
    return float(_CL_ARC[idx] / _CL_TOTAL_LEN)


def track_tangent(x: float, y: float) -> np.ndarray:
    """
    Unit tangent vector of the nearest centerline segment at (x, y).
    Points in the direction of travel (increasing lap progress).
    """
    ax_ = _CL_X[:-1];  ay_ = _CL_Y[:-1]
    bx_ = _CL_X[1:];   by_ = _CL_Y[1:]
    dx  = bx_ - ax_;   dy  = by_ - ay_
    den = np.maximum(dx*dx + dy*dy, 1e-12)
    t   = np.clip(((x - ax_)*dx + (y - ay_)*dy) / den, 0.0, 1.0)
    d2  = (ax_ + t*dx - x)**2 + (ay_ + t*dy - y)**2
    idx = int(np.argmin(d2))
    return np.array([_CL_TX[idx], _CL_TY[idx]], dtype=np.float64)


calcular_tangente_pista = track_tangent


def nearest_centerline_index(x: float, y: float) -> int:
    """Index of the nearest centerline vertex to (x, y)."""
    d2 = (_CENTERLINE[:, 0] - x)**2 + (_CENTERLINE[:, 1] - y)**2
    return int(np.argmin(d2))


# ─────────────────────────────────────────────────────────────
#  TRACK BORDER GENERATION
# ─────────────────────────────────────────────────────────────
def generate_track_borders() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute left/right offset curves at ±half-width from centerline.
    Returns (outer_x, outer_y, inner_x, inner_y).
    """
    hw = _TRACK_WIDTH / 2.0
    n  = len(_CENTERLINE)
    tx = np.zeros(n, dtype=np.float64)
    ty = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d   = _CENTERLINE[(i + 1) % n] - _CENTERLINE[(i - 1) % n]
        mag = np.sqrt(d[0]**2 + d[1]**2)
        if mag > 1e-12:
            tx[i], ty[i] = d[0] / mag, d[1] / mag
        else:
            tx[i], ty[i] = 1.0, 0.0

    ox = _CENTERLINE[:, 0] + hw * ty;  oy = _CENTERLINE[:, 1] - hw * tx
    ix = _CENTERLINE[:, 0] - hw * ty;  iy = _CENTERLINE[:, 1] + hw * tx
    return ox, oy, ix, iy


gerar_contornos_pista = generate_track_borders


# ─────────────────────────────────────────────────────────────
#  FINISH LINE CROSSING
# ─────────────────────────────────────────────────────────────
def crosses_finish_line(x0: float, y0: float, x1: float, y1: float) -> bool:
    """
    True if the segment (x0,y0)→(x1,y1) crosses the finish line.
    Uses a 2-D segment intersection in the finish line's local frame.
    """
    fl      = CONFIG['finish_line']
    cx, cy  = fl['x'], fl['y']
    rot     = -np.radians(fl['angle'])
    cos_r   = np.cos(rot);  sin_r = np.sin(rot)

    def _local(px, py):
        rx, ry = px - cx, py - cy
        return rx*cos_r - ry*sin_r, rx*sin_r + ry*cos_r

    x0l, y0l = _local(x0, y0)
    x1l, y1l = _local(x1, y1)


    if y0l * y1l > 0 or y0l == y1l:
        return False

    t       = y0l / (y0l - y1l)
    x_cross = x0l + t * (x1l - x0l)
    return bool(abs(x_cross) <= fl['width'] / 2.0)


cruza_linha_chegada = crosses_finish_line


# ─────────────────────────────────────────────────────────────
#  CAR SPAWN POSITIONS
# ─────────────────────────────────────────────────────────────
def spawn_position(car_index: int, total_cars: int
                   ) -> Tuple[float, float, float]:
    """
    Returns (x, y, angle_deg) for the car's starting position,
    spread evenly across 80 % of the start-line width.
    """
    sl  = CONFIG['start_line']
    cx, cy  = sl['x'], sl['y']
    span    = sl['width'] * 0.80

    t = 0.0 if total_cars <= 1 else (car_index / (total_cars - 1) - 0.5)
    x_loc = t * span;  y_loc = 0.0

    rot   = np.radians(sl['angle'])
    cos_r = np.cos(rot);  sin_r = np.sin(rot)
    return (cx + x_loc*cos_r - y_loc*sin_r,
            cy + x_loc*sin_r + y_loc*cos_r,
            CONFIG['cars']['initial_angle'])


calcular_posicao_inicial_carro = spawn_position


# ─────────────────────────────────────────────────────────────
#  CAR ENTITY
# ─────────────────────────────────────────────────────────────
class AICar:
    """
    Neural-network-controlled car entity.

    Attributes
    ----------
    x, y          : float  — current world position
    angle         : float  — heading in degrees
    speed         : float  — current speed (world units / frame)
    alive         : bool
    laps          : int    — completed laps
    score         : float  — accumulated fitness score
    death_reason  : str | None
    last_action   : ndarray([steer, throttle])
    frame_state   : dict   — per-frame event flags for the renderer
    events        : deque  — recent event strings for the event log
    """

    # Sensor angles relative to car heading (degrees)
    _SENSOR_ANGLES = np.array([-90., -45., -22.5, 0., 22.5, 45., 90.],
                               dtype=np.float32)
    _SENSOR_MAX    = 15.0   # world units
    _SENSOR_STEPS  = np.arange(0.2, _SENSOR_MAX + 0.01, 0.2, dtype=np.float32)
    _N_STEPS       = len(_SENSOR_STEPS)

    def __init__(self, car_index: int = 0, total_cars: int = 1):
        self.car_index  = car_index
        self.total_cars = total_cars
        self.reset()

    # ── Public interface ──────────────────────────────────────
    def reset(self):
        x, y, angle = spawn_position(self.car_index, self.total_cars)
        cfg_cars = CONFIG['cars']
        cfg_pen  = CONFIG['penalties']
        cfg_sim  = CONFIG['simulation']

        self.x        = x;   self.y     = y
        self.speed    = (cfg_cars['max_speed']
                         * cfg_cars['initial_speed_pct'] / 100.0)
        self.angle    = float(angle)
        self.alive    = True

        # Timing
        self.frame_alive        = 0
        self.laps               = 0
        self.lap_start_frame    = 0
        self.best_lap_frames: Optional[int] = None

        # Scoring
        self.score              = 0.0
        self.total_speed        = 0.0
        self.death_reason: Optional[str] = None

        # Lap validation
        self.max_finish_dist    = 0.0
        self.cl_checkpoint      = 0
        self.checkpoint_frames  = int(np.random.randint(
            0, cfg_sim['frames_without_progress']))
        self.max_cl_index       = 0
        self.cl_lap_start_idx   = 0

        # Trailing state
        self._score_window = deque(maxlen=cfg_pen['continuous_loss_window'])
        self.last_action   = np.array([0.0, 0.5], dtype=np.float32)
        self._frame_state  = {}   
        self.events        = deque(maxlen=12)
        self._cached_sensors: Optional[np.ndarray] = None

        # Public trajectory (kept short for rendering)
        self.traj_x: list = [self.x]
        self.traj_y: list = [self.y]

    def step(self, action: np.ndarray):
        """Advance physics, check collisions, update score. Called each sim frame."""
        if not self.alive:
            return
        self.last_action = action
        self.frame_state = {}
        prev_x, prev_y = self.x, self.y

        self._apply_physics(action)
        self._check_lap(prev_x, prev_y)
        self._score_speed()
        self._check_wrong_way()
        self._check_near_wall()
        self._check_continuous_loss()

    def get_sensors(self) -> np.ndarray:
        """
        Returns 8 floats: 7 normalised distance sensors + normalised speed.
        Values in [0, 1].  Cached per frame.
        """
        rads  = np.radians(self.angle + self._SENSOR_ANGLES)
        steps = self._SENSOR_STEPS

        sx = np.float32(self.x) + np.cos(rads)[:, None] * steps   
        sy = np.float32(self.y) + np.sin(rads)[:, None] * steps

        inside = points_on_track_batch(sx.ravel(), sy.ravel()).reshape(7, self._N_STEPS)
        dists  = np.full(7, self._SENSOR_MAX, dtype=np.float32)
        off    = ~inside
        hit    = off.any(axis=1)
        if hit.any():
            first = np.argmax(off, axis=1)
            dists[hit] = steps[first[hit]]

        vel_norm = self.speed / CONFIG['cars']['max_speed']
        sensors  = np.append(dists / self._SENSOR_MAX, vel_norm).astype(np.float32)
        self._cached_sensors = sensors
        return sensors

    def check_collision(self) -> bool:
        """Kill car if outside track. Returns True if just died."""
        if not point_on_track(self.x, self.y):
            self.alive        = False
            self.death_reason = 'Off track'
            self.score       -= CONFIG['penalties']['collision_penalty']
            self.events.append(('CRASH!', _COR['red']))
            return True
        return False

    # ── Physics ──────────────────────────────────────────────
    def _apply_physics(self, action: np.ndarray):
        cfg  = CONFIG['cars']
        vmax = cfg['max_speed'];  step = cfg['speed_step']

        # Steering: action[0] in [-1, +1]
        self.angle += float(action[0]) * cfg['max_turn_angle']

        # Throttle: action[1] ≥ 0.5 → accelerate
        if float(action[1]) >= 0.5:
            self.speed = min(vmax, self.speed + step)
        else:
            self.speed = max(0.0, self.speed - step)

        rad     = np.radians(self.angle)
        self.x += self.speed * np.cos(rad)
        self.y += self.speed * np.sin(rad)

        self.total_speed  += self.speed
        self.frame_alive  += 1

        # Finish-line distance (needed for valid-lap detection)
        fl   = CONFIG['finish_line']
        dist = np.hypot(self.x - fl['x'], self.y - fl['y'])
        if dist > self.max_finish_dist:
            self.max_finish_dist = dist

        # Nearest centerline index (progress tracking)
        cl_idx = nearest_centerline_index(self.x, self.y)
        if cl_idx > self.max_cl_index:
            self.max_cl_index = cl_idx
            self.checkpoint_frames = 0
        else:
            self.checkpoint_frames += 1

        # Trajectory buffer (capped for performance)
        self.traj_x.append(self.x);  self.traj_y.append(self.y)
        if len(self.traj_x) > 300:
            self.traj_x = self.traj_x[-200:]
            self.traj_y = self.traj_y[-200:]

        # Stagnation kill
        cfg_sim = CONFIG['simulation']
        if self.checkpoint_frames > cfg_sim['frames_without_progress']:
            self.alive        = False
            self.death_reason = 'No progress'
            self.events.append(('No progress', _COR['red']))

    # ── Lap detection ─────────────────────────────────────────
    def _check_lap(self, prev_x: float, prev_y: float):
        if not crosses_finish_line(prev_x, prev_y, self.x, self.y):
            return

        cfg_det = CONFIG['lap_detection']
        min_progress = len(_CENTERLINE) * cfg_det['min_lap_fraction']
        real_progress = self.max_cl_index - self.cl_lap_start_idx

        if (self.max_finish_dist < cfg_det['min_distance_from_finish']
                or real_progress < min_progress):
            return

        lap_time   = self.frame_alive - self.lap_start_frame
        is_record  = (self.best_lap_frames is None or lap_time < self.best_lap_frames)
        improvement = ((self.best_lap_frames - lap_time)
                       if (is_record and self.best_lap_frames) else 0)

        if is_record:
            self.best_lap_frames = lap_time

        # Reset lap state
        self.lap_start_frame    = self.frame_alive
        self.laps              += 1
        self.max_finish_dist    = 0.0
        self.cl_lap_start_idx   = self.max_cl_index
        self.cl_checkpoint      = self.max_cl_index
        self.checkpoint_frames  = 0
        self.frame_state['lap'] = True

        # Lap reward
        rec = CONFIG['rewards']
        self.score += rec['lap_reward']
        self.events.append((f"+{rec['lap_reward']:.0f} LAP!", _COR['green']))

        # Speed bonus
        speed_bonus = rec['fast_lap_bonus'] * rec['target_lap_frames'] / max(lap_time, 1)
        self.score += speed_bonus
        self.events.append((f'+{speed_bonus:.0f} spd ({lap_time}fr)', _COR['green']))

        # Record bonus
        if improvement > 0:
            old_time = lap_time + improvement
            rec_bonus = (rec['lap_reward']
                         * (improvement / old_time)
                         * self.laps
                         * rec['record_bonus_factor'])
            self.score += rec_bonus
            self.events.append((f'+{rec_bonus:.0f} RECORD!', _COR['yellow']))

    # ── Speed reward / penalty ────────────────────────────────
    def _score_speed(self):
        if self.laps == 0:
            self.frame_state['fast'] = True
            return

        pen   = CONFIG['penalties']
        rec   = CONFIG['rewards']
        v_pct = self.speed / CONFIG['cars']['max_speed']

        threshold  = min(0.9, pen['initial_slow_threshold']
                         + (self.laps - 1) * pen['slow_threshold_increment'])
        weight_spd = rec['speed_weight']     + (self.laps - 1) * pen['aggravator_per_lap']
        weight_slw = pen['slow_weight_post_lap'] + (self.laps - 1) * pen['aggravator_per_lap']

        if v_pct >= threshold:
            pts = (v_pct - threshold) / max(1.0 - threshold, 1e-6) * weight_spd
            self.frame_state['fast'] = True
        else:
            pts = -((threshold - v_pct) / max(threshold, 1e-6)) * weight_slw
            self.frame_state['slow'] = True

        self.score += pts

    # ── Wrong-way penalty ─────────────────────────────────────
    def _check_wrong_way(self):
        tangent   = track_tangent(self.x, self.y)
        rad       = np.radians(self.angle)
        car_dir   = np.array([np.cos(rad), np.sin(rad)])
        dot       = float(np.dot(car_dir, tangent))
        threshold = CONFIG['penalties']['wrong_way_threshold']

        if dot < threshold:
            pen = CONFIG['penalties']['wrong_way_penalty']
            if self.laps == 0:

                self.speed = 0.0
            else:
                self.score -= pen
            self.frame_state['wrong_way'] = True

    # ── Near-wall penalty ─────────────────────────────────────
    def _check_near_wall(self):
        if self.laps == 0:
            return  

        dist_to_cl = distance_to_centerline(self.x, self.y)
        hw         = _TRACK_WIDTH / 2.0

        proximity  = (dist_to_cl - hw * 0.75) / (hw * 0.25)
        if proximity > 0:
            pen = CONFIG['penalties']['near_wall_penalty'] * min(proximity, 1.0)
            self.score -= pen
            self.frame_state['wall'] = True

    # ── Continuous loss kill ──────────────────────────────────
    def _check_continuous_loss(self):
        self._score_window.append(self.score)
        if (self.frame_alive <= self._score_window.maxlen
                or len(self._score_window) < self._score_window.maxlen):
            return

        loss      = self._score_window[-1] - self._score_window[0]
        pen       = CONFIG['penalties']
        threshold = (pen['continuous_loss_threshold_pre_lap']
                     if self.laps == 0
                     else pen['continuous_loss_threshold_post_lap'])

        if loss < threshold and self.score < pen['min_score_kill']:
            self.alive        = False
            self.death_reason = 'Continuous loss'
            self.events.append(('Cont. loss', _COR['red']))

    # ── Legacy property aliases (keep renderer compatibility) ─
    @property
    def vivo(self):        return self.alive
    @vivo.setter
    def vivo(self, v):     self.alive = v

    @property
    def velocidade(self):        return self.speed
    @velocidade.setter
    def velocidade(self, v):     self.speed = v

    @property
    def voltas_completas(self):       return self.laps
    @voltas_completas.setter
    def voltas_completas(self, v):    self.laps = v

    @property
    def pontos_acumulados(self):       return self.score
    @pontos_acumulados.setter
    def pontos_acumulados(self, v):    self.score = v

    @property
    def tempo_vivo(self):        return self.frame_alive
    @tempo_vivo.setter
    def tempo_vivo(self, v):     self.frame_alive = v

    @property
    def ultima_acao(self):        return self.last_action
    @ultima_acao.setter
    def ultima_acao(self, v):     self.last_action = v

    @property
    def estado_frame(self):        return self._frame_state
    @estado_frame.setter
    def estado_frame(self, v):     self._frame_state = v

    @property
    def melhor_tempo_volta(self):        return self.best_lap_frames
    @melhor_tempo_volta.setter
    def melhor_tempo_volta(self, v):     self.best_lap_frames = v

    @property
    def motivo_morte(self):        return self.death_reason
    @motivo_morte.setter
    def motivo_morte(self, v):     self.death_reason = v

    @property
    def frame_inicio_volta(self):        return self.lap_start_frame
    @frame_inicio_volta.setter
    def frame_inicio_volta(self, v):     self.lap_start_frame = v

    @property
    def angulo(self):        return self.angle
    @angulo.setter
    def angulo(self, v):     self.angle = v

    # Keep old method names
    def mover(self, action):    self.step(action)
    def checar_colisao(self):   return self.check_collision()
    def get_sensores(self):     return self.get_sensors()

    def __repr__(self):
        return (f'<AICar #{self.car_index}  laps={self.laps}  '
                f'score={self.score:.1f}  alive={self.alive}>')



CarrinhoIA = AICar