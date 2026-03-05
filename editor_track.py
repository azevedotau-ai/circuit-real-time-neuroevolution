# Interactive track editor — click to add points, generates Catmull-Rom spline.
# Exports track.json for use in the simulator.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.patches import FancyArrowPatch, Rectangle, PathPatch, FancyBboxPatch, Circle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import json, os, math
try:
    import tkinter as tk
except ImportError:
    tk = None

# ──────────────────────────────────────────
#  CONFIG LOADER
# ──────────────────────────────────────────
def _load_config():
    cp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setting.json')
    if os.path.exists(cp):
        with open(cp, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

_cfg = _load_config()
_C = _cfg.get('ui_colors', {})

# Fallback palette (full dark-racing theme)
PAL = {
    'bg':           _C.get('black',          '#0a0d12'),
    'asphalt':      _C.get('asphalt_color',  '#1c2533'),
    'grass':        _C.get('grass_color',    '#0d1520'),
    'card':         _C.get('card_color',     '#151c27'),
    'border':                                '#1e2d42',
    'white':        _C.get('white',          '#e8ecf1'),
    'green':        _C.get('green',          '#7BFF00'),
    'yellow':       _C.get('yellow',         '#c9a84c'),
    'red':          _C.get('red',            '#FD2C2C'),
    'cyan':                                  '#00d4e8',
    'inactive':     _C.get('inactive_gray',  '#2e3a47'),
    'mid_gray':     _C.get('medium_gray',    '#5a6a7a'),
    'text_primary': _C.get('primary_text',   '#dce3ea'),
    'text_secondary':_C.get('secondary_text','#8fa0b0'),
    'cp_color':     _C.get('checkpoint_color','#c9a84c'),
    'normal_car':   _C.get('normal_color',   '#488E97'),
    'best_car':     _C.get('best_color',     '#7BFF00'),
}

TRACK_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'track.json')
SNAP_R = 0.9       # snap/remove radius
N_SEG  = 50        # spline segments per control-point pair (higher = smoother)


# ──────────────────────────────────────────
#  MATH HELPERS
# ──────────────────────────────────────────
def catmull_rom(pts: np.ndarray, n_seg: int = N_SEG) -> np.ndarray:
    """Closed Catmull-Rom spline through control points."""
    closed = np.vstack([pts, pts[0]])
    n = len(closed) - 1
    ts  = np.linspace(0, 1, n_seg, endpoint=False)
    tt  = ts * ts
    ttt = tt * ts
    out = []
    for i in range(n):
        p0 = closed[i - 1] if i > 0 else closed[n - 1]
        p1 = closed[i]
        p2 = closed[i + 1]
        p3 = closed[i + 2] if i + 2 <= n else closed[(i + 2) - n]
        seg = 0.5 * (
            np.outer(-ttt + 2*tt - ts,    p0) +
            np.outer( 3*ttt - 5*tt + 2,   p1) +
            np.outer(-3*ttt + 4*tt + ts,   p2) +
            np.outer( ttt - tt,            p3)
        )
        out.append(seg)
    return np.vstack(out)


def track_borders(cl: np.ndarray, hw: float):
    """Left/right offset curves at ±hw from centerline."""
    n = len(cl)
    cl_c = np.vstack([cl, cl[0]])
    tx = np.zeros(n); ty = np.zeros(n)
    for i in range(n):
        d = cl_c[(i + 1) % (n + 1)] - cl_c[(i - 1) % n]
        mag = np.hypot(d[0], d[1])
        tx[i], ty[i] = (d / mag) if mag > 1e-12 else (1.0, 0.0)
    ox = cl[:, 0] + hw * ty;  oy = cl[:, 1] - hw * tx
    ix = cl[:, 0] - hw * ty;  iy = cl[:, 1] + hw * tx
    return ox, oy, ix, iy


def track_length(cl: np.ndarray) -> float:
    diff = np.diff(np.vstack([cl, cl[0]]), axis=0)
    return float(np.sum(np.hypot(diff[:, 0], diff[:, 1])))


def curvature(cl: np.ndarray) -> np.ndarray:
    """Approximate signed curvature at each spline point."""
    n = len(cl)
    kappa = np.zeros(n)
    for i in range(n):
        a = cl[(i - 1) % n]; b = cl[i]; c = cl[(i + 1) % n]
        ab = b - a; bc = c - b
        cross = ab[0]*bc[1] - ab[1]*bc[0]
        denom = (np.hypot(*ab) * np.hypot(*bc) * np.hypot(*(c - a))) + 1e-12
        kappa[i] = 2 * cross / denom
    return kappa


def gradient_segments(x: np.ndarray, y: np.ndarray,
                       values: np.ndarray, cmap, vmin=None, vmax=None):
    """Build a LineCollection coloured by `values` along the path."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.5, zorder=5)
    lc.set_array(values[:-1])
    return lc


# ──────────────────────────────────────────
#  STYLED WIDGET HELPERS
# ──────────────────────────────────────────
def _style_button(btn, fg_color, bg='#1a2535', hover=None):
    btn.ax.set_facecolor(bg)
    btn.color       = bg
    btn.hovercolor  = hover or PAL['inactive']
    btn.label.set_color(fg_color)
    btn.label.set_fontsize(8.5)
    btn.label.set_fontweight('bold')
    btn.label.set_fontfamily('monospace')
    for sp in btn.ax.spines.values():
        sp.set_edgecolor(PAL['border'])
        sp.set_linewidth(1.2)


def _panel_bg(fig, rect, color=None, alpha=0.85, radius=0.02):
    """Draw a rounded rectangle panel on the figure (axes-independent)."""
    color = color or PAL['card']
    ax_bg = fig.add_axes(rect, zorder=0)
    ax_bg.set_facecolor(color)
    ax_bg.set_alpha(alpha)
    ax_bg.set_axis_off()
    for sp in ax_bg.spines.values():
        sp.set_visible(False)
    return ax_bg


def _section_label(fig, x, y, text, color=None):
    fig.text(x, y, text,
             fontsize=6.5, fontweight='bold', color=color or PAL['text_secondary'],
             va='bottom', fontfamily='monospace', alpha=0.9)


# ──────────────────────────────────────────
#  MINIMAP AXIS
# ──────────────────────────────────────────
class Minimap:
    """Small overview panel showing the full track + curvature heat."""
    def __init__(self, fig, rect):
        self.ax = fig.add_axes(rect)
        self.ax.set_facecolor(PAL['bg'])
        self.ax.set_axis_off()
        self.ax.set_title('OVERVIEW', color=PAL['text_secondary'],
                          fontsize=6, fontfamily='monospace', pad=3)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(PAL['border'])
            sp.set_linewidth(0.8)
            sp.set_visible(True)
        self._artists = []

    def update(self, cl=None, hw=None, pts=None):
        for a in self._artists:
            try: a.remove()
            except: pass
        self._artists = []
        ax = self.ax
        ax.set_facecolor(PAL['bg'])

        if cl is None or len(cl) < 4:
            t = ax.text(0.5, 0.5, 'ADD ≥3 POINTS',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=6, color=PAL['text_secondary'], fontfamily='monospace')
            self._artists.append(t)
            return

        # Curvature-coloured centerline
        kap = np.abs(curvature(cl))
        cx = np.append(cl[:, 0], cl[0, 0])
        cy = np.append(cl[:, 1], cl[0, 1])
        kv = np.append(kap, kap[0])
        lc = gradient_segments(cx, cy, kv, plt.cm.RdYlGn_r, 0, kap.max() + 1e-9)
        ax.add_collection(lc)
        self._artists.append(lc)

        # Borders
        ox, oy, ix, iy = track_borders(cl, hw)
        for bx, by in [(ox, oy), (ix, iy)]:
            l, = ax.plot(np.append(bx, bx[0]), np.append(by, by[0]),
                         color=PAL['white'], lw=0.6, alpha=0.4)
            self._artists.append(l)

        # Control points
        if pts is not None and len(pts):
            pa = np.array(pts)
            sc = ax.scatter(pa[:, 0], pa[:, 1], s=14,
                            color=PAL['cp_color'], edgecolors=PAL['bg'],
                            linewidths=0.5, zorder=6)
            self._artists.append(sc)

        margin = hw * 2
        ax.set_xlim(cx.min() - margin, cx.max() + margin)
        ax.set_ylim(cy.min() - margin, cy.max() + margin)
        ax.set_aspect('equal')


# ──────────────────────────────────────────
#  STATS PANEL
# ──────────────────────────────────────────
class StatsPanel:
    """Compact stats display rendered as text in a figure axes."""
    def __init__(self, fig, rect):
        self.ax = fig.add_axes(rect)
        self.ax.set_facecolor(PAL['card'])
        self.ax.set_axis_off()
        self.ax.set_title('TRACK STATS', color=PAL['text_secondary'],
                          fontsize=6, fontfamily='monospace', pad=3)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(PAL['border']); sp.set_linewidth(0.8); sp.set_visible(True)
        self._texts = []

    def update(self, cl=None, hw=None, n_pts=0, name=''):
        for t in self._texts:
            try: t.remove()
            except: pass
        self._texts = []
        ax = self.ax

        if cl is not None and len(cl) >= 4:
            length    = track_length(cl)
            kap       = np.abs(curvature(cl))
            max_kap   = float(kap.max())
            mean_kap  = float(kap.mean())
            difficulty = min(10.0, (mean_kap * 1200 + max_kap * 400))
            diff_color = PAL['green'] if difficulty < 4 else PAL['yellow'] if difficulty < 7 else PAL['red']

            rows = [
                ('POINTS',     f'{n_pts}',           PAL['text_primary']),
                ('WIDTH',      f'{hw*2:.2f} u',       PAL['cyan']),
                ('LENGTH',     f'{length:.1f} u',     PAL['cyan']),
                ('MAX CURVE',  f'{max_kap:.4f}',      PAL['yellow']),
                ('AVG CURVE',  f'{mean_kap:.4f}',     PAL['yellow']),
                ('DIFFICULTY', f'{difficulty:.1f}/10', diff_color),
            ]
        else:
            rows = [
                ('POINTS',    f'{n_pts}',  PAL['text_primary']),
                ('WIDTH',     '—',         PAL['text_secondary']),
                ('LENGTH',    '—',         PAL['text_secondary']),
                ('MAX CURVE', '—',         PAL['text_secondary']),
                ('AVG CURVE', '—',         PAL['text_secondary']),
                ('DIFFICULTY','—',         PAL['text_secondary']),
            ]

        for i, (label, value, color) in enumerate(rows):
            y = 0.88 - i * 0.155
            tl = ax.text(0.07, y, label, transform=ax.transAxes,
                         fontsize=6.5, color=PAL['text_secondary'],
                         fontfamily='monospace', va='top')
            tv = ax.text(0.93, y, value, transform=ax.transAxes,
                         fontsize=7.5, color=color, fontfamily='monospace',
                         va='top', ha='right', fontweight='bold')
            self._texts += [tl, tv]

            # Separator line
            if i < len(rows) - 1:
                sep_y = y - 0.09
                ln = ax.axhline(sep_y, xmin=0.05, xmax=0.95,
                                color=PAL['border'], linewidth=0.5,
                                transform=ax.transAxes)
                self._texts.append(ln)


# ──────────────────────────────────────────
#  CURVATURE CHART
# ──────────────────────────────────────────
class CurvatureChart:
    """Small curvature profile chart."""
    def __init__(self, fig, rect):
        self.ax = fig.add_axes(rect)
        self.ax.set_facecolor(PAL['card'])
        self.ax.set_title('CURVATURE PROFILE', color=PAL['text_secondary'],
                          fontsize=6, fontfamily='monospace', pad=3)
        self.ax.tick_params(labelsize=5, colors=PAL['text_secondary'])
        for sp in self.ax.spines.values():
            sp.set_edgecolor(PAL['border']); sp.set_linewidth(0.8)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_facecolor(PAL['card'])
        self._fill = None
        self._line = None

    def update(self, cl=None):
        if self._fill:
            try: self._fill.remove()
            except: pass
            self._fill = None
        if self._line:
            try: self._line.remove()
            except: pass
            self._line = None

        ax = self.ax
        ax.cla()
        ax.set_facecolor(PAL['card'])
        ax.set_title('CURVATURE PROFILE', color=PAL['text_secondary'],
                     fontsize=6, fontfamily='monospace', pad=3)
        ax.tick_params(labelsize=5, colors=PAL['text_secondary'])
        for sp in ax.spines.values():
            sp.set_edgecolor(PAL['border']); sp.set_linewidth(0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if cl is None or len(cl) < 4:
            ax.text(0.5, 0.5, 'NO DATA', transform=ax.transAxes,
                    ha='center', va='center', fontsize=6,
                    color=PAL['text_secondary'], fontfamily='monospace')
            return

        kap = np.abs(curvature(cl))
        x   = np.linspace(0, 100, len(kap))
        ax.fill_between(x, kap, alpha=0.25, color=PAL['cyan'])
        ax.plot(x, kap, color=PAL['cyan'], lw=1.0)
        ax.axhline(kap.mean(), color=PAL['yellow'], lw=0.7, ls='--', alpha=0.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, kap.max() * 1.3 + 1e-9)
        ax.set_xlabel('% lap', fontsize=5, color=PAL['text_secondary'],
                      fontfamily='monospace')
        ax.set_ylabel('|κ|',   fontsize=5, color=PAL['text_secondary'],
                      fontfamily='monospace')


# ──────────────────────────────────────────
#  MAIN EDITOR
# ──────────────────────────────────────────
class TrackEditor:
    """
    Advanced interactive track editor with:
      • Catmull-Rom spline preview with curvature heatmap
      • Drag & drop control points
      • Minimap, stats panel, curvature chart
      • Undo stack, snap-to-grid toggle
      • Export to track.json
    """

    def __init__(self):
        self.points: list[list[float]] = []
        self.width      = 1.8
        self.track_name = 'Track 1'
        self._drag_idx  = None
        self._undo_stack: list[tuple] = []   # list of (points_copy, width)
        self._snap_grid  = False
        self._grid_size  = 1.0
        self._show_cp_labels = True
        self._show_curv_heat = True
        self._hovered_idx    = None

        self._load_track_json()
        self._build_figure()
        self._redraw()
        plt.show()

    # ── PERSISTENCE ──────────────────────────────
    def _load_track_json(self):
        if not os.path.exists(TRACK_JSON):
            return
        try:
            with open(TRACK_JSON, 'r', encoding='utf-8') as f:
                d = json.load(f)
            pts = d.get('control_points', d.get('pontos_controle', []))
            if len(pts) >= 3:
                arr = np.array(pts)
                if np.allclose(arr[0], arr[-1]):
                    arr = arr[:-1]
                self.points = arr.tolist()
            self.width      = float(d.get('track_width', d.get('largura_pista', 1.8)))
            self.track_name = d.get('name', d.get('nome', 'Track 1'))
            print(f'[editor] Loaded "{self.track_name}" — {len(self.points)} pts, w={self.width:.2f}')
        except Exception as e:
            print(f'[editor] Could not load track: {e}')

    def _save_track_json(self, _=None):
        n = len(self.points)
        if n < 3:
            print('[editor] Need ≥ 3 points to save.')
            return

        pts_arr = np.array(self.points)
        cl      = catmull_rom(pts_arr)
        hw      = self.width / 2.0

        mid_raw = (pts_arr[0] + pts_arr[1]) / 2.0
        dists   = np.hypot(cl[:, 0] - mid_raw[0], cl[:, 1] - mid_raw[1])
        idx_cl  = int(np.argmin(dists))
        prev_cl = cl[(idx_cl - 1) % len(cl)]
        next_cl = cl[(idx_cl + 1) % len(cl)]
        tang    = next_cl - prev_cl
        mid     = cl[idx_cl]
        car_ang = float(np.degrees(np.arctan2(tang[1], tang[0])))
        line_ang = car_ang + 90.0

        closed_pts = pts_arr.tolist()
        closed_pts.append(closed_pts[0])

        data = {
            'name':           self.track_name,
            'control_points': closed_pts,
            'track_width':    float(self.width),
            'start': {
                'x': float(mid[0]),  'y': float(mid[1]),
                'line_width': float(self.width),
                'car_angle_deg':  float(car_ang),
                'line_angle_deg': float(line_ang),
            },
            'finish': {
                'x': float(mid[0]),  'y': float(mid[1]),
                'line_width': float(self.width),
                'line_angle_deg': float(line_ang),
            },
        }
        with open(TRACK_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        length = track_length(cl)
        print(f'[editor] Saved "{self.track_name}" → {TRACK_JSON}')
        print(f'         {n} pts | width {self.width:.2f} | length {length:.1f} | dir {car_ang:.0f}°')

    # ── UNDO ─────────────────────────────────────
    def _push_undo(self):
        self._undo_stack.append(([p[:] for p in self.points], self.width))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _undo(self, _=None):
        if not self._undo_stack:
            return
        pts, w = self._undo_stack.pop()
        self.points = pts
        self.width  = w
        self.sl_width.set_val(w)
        self._redraw()

    def _clear(self, _=None):
        self._push_undo()
        self.points.clear()
        self._redraw()

    # ── FIGURE BUILD ─────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(21, 11.5))
        self.fig.patch.set_facecolor(PAL['bg'])

        # ── Right sidebar: 3 stacked panels ──────
        # Positions: [left, bottom, width, height]
        _rp_l = 0.775
        _rp_w = 0.215
        self.minimap    = Minimap(self.fig,       [_rp_l, 0.565, _rp_w, 0.400])
        self.stats_panel = StatsPanel(self.fig,   [_rp_l, 0.240, _rp_w, 0.305])
        self.curv_chart  = CurvatureChart(self.fig,[_rp_l, 0.140, _rp_w, 0.090])

        # ── Main canvas ──────────────────────────
        self.ax = self.fig.add_axes([0.025, 0.145, 0.735, 0.830])
        self._config_canvas()

        # ── Bottom toolbar ────────────────────────
        _bot_y   = 0.020
        _bot_h   = 0.085
        _sep_y   = 0.118
        _lbl_y   = _sep_y + 0.016

        # Separator line
        ax_sep = self.fig.add_axes([0.0, _sep_y, 1.0, 0.001])
        ax_sep.set_facecolor(PAL['border']); ax_sep.set_axis_off()

        # Section labels
        _section_label(self.fig, 0.026, _lbl_y, '▸ TRACK NAME')
        _section_label(self.fig, 0.240, _lbl_y, '▸ ACTIONS')
        _section_label(self.fig, 0.530, _lbl_y, '▸ TRACK WIDTH')
        _section_label(self.fig, 0.775, _lbl_y, '▸ OPTIONS')

        # TextBox — track name
        ax_name = self.fig.add_axes([0.026, _bot_y, 0.190, _bot_h])
        self.txt_name = TextBox(ax_name, '',
                                initial=self.track_name,
                                color=PAL['inactive'],
                                hovercolor='#243040')
        self.txt_name.text_disp.set_color(PAL['white'])
        self.txt_name.text_disp.set_fontsize(9)
        self.txt_name.text_disp.set_fontfamily('monospace')
        self.txt_name.text_disp.set_fontweight('bold')
        for sp in ax_name.spines.values():
            sp.set_edgecolor(PAL['border']); sp.set_linewidth(1.3)
        self.txt_name.on_text_change(self._cb_name)

        # Action buttons
        _bw = 0.068; _gap = 0.007; _bx = 0.240
        ax_save  = self.fig.add_axes([_bx,                    _bot_y, _bw, _bot_h])
        ax_undo  = self.fig.add_axes([_bx + (_bw+_gap),       _bot_y, _bw, _bot_h])
        ax_clear = self.fig.add_axes([_bx + 2*(_bw+_gap),     _bot_y, _bw, _bot_h])

        self.btn_save  = Button(ax_save,  '💾  SAVE',  color='#0f2a10')
        self.btn_undo  = Button(ax_undo,  '↩  UNDO',  color='#1a2535')
        self.btn_clear = Button(ax_clear, '✕  CLEAR', color='#2a0f0f')

        _style_button(self.btn_save,  PAL['green'],  '#0f2a10', '#1a4a1c')
        _style_button(self.btn_undo,  PAL['white'],  '#1a2535', '#2e3f52')
        _style_button(self.btn_clear, PAL['red'],    '#2a0f0f', '#4a1515')

        # Width slider
        ax_sl = self.fig.add_axes([0.530, _bot_y + 0.022, 0.225, 0.038])
        self.sl_width = Slider(ax_sl, '', 0.4, 6.0,
                               valinit=self.width, color=PAL['cyan'])
        self.sl_width.poly.set_alpha(0.7)
        self.sl_width.valtext.set_color(PAL['white'])
        self.sl_width.valtext.set_fontsize(9)
        self.sl_width.valtext.set_fontfamily('monospace')
        ax_sl.set_facecolor(PAL['inactive'])
        for sp in ax_sl.spines.values():
            sp.set_edgecolor(PAL['border']); sp.set_linewidth(1.1)

        # Option toggle buttons
        _ox = 0.775; _ow = 0.100
        ax_snap  = self.fig.add_axes([_ox,           _bot_y, _ow, _bot_h])
        ax_label = self.fig.add_axes([_ox+_ow+0.005, _bot_y, _ow, _bot_h])

        self.btn_snap  = Button(ax_snap,  'SNAP OFF', color=PAL['inactive'])
        self.btn_label = Button(ax_label, 'LABELS ON',color='#0a1e2e')
        _style_button(self.btn_snap,  PAL['text_secondary'], PAL['inactive'])
        _style_button(self.btn_label, PAL['cyan'], '#0a1e2e')

        # Wire callbacks
        self.btn_save.on_clicked(self._save_track_json)
        self.btn_undo.on_clicked(self._undo)
        self.btn_clear.on_clicked(self._clear)
        self.sl_width.on_changed(self._cb_width)
        self.btn_snap.on_clicked(self._toggle_snap)
        self.btn_label.on_clicked(self._toggle_labels)

        self.fig.canvas.mpl_connect('key_press_event',      self._cb_key)
        self.fig.canvas.mpl_connect('button_press_event',   self._cb_press)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._cb_move)
        self.fig.canvas.mpl_connect('button_release_event', self._cb_release)

    # ── CANVAS CONFIG ────────────────────────────
    def _config_canvas(self):
        ax = self.ax
        ax.set_facecolor(PAL['grass'])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color=PAL['border'], alpha=0.5, linewidth=0.5, linestyle=':')
        ax.tick_params(colors=PAL['mid_gray'], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(PAL['border']); sp.set_linewidth(1.2)

        # Styled title
        ax.set_title(
            'TRACK EDITOR  ·  Left-click: add / drag   ·   Right-click: remove   ·   Ctrl+Z: undo',
            color=PAL['text_secondary'], fontsize=9.5, pad=10,
            fontfamily='monospace', fontweight='bold'
        )

    # ── CALLBACKS ────────────────────────────────
    def _cb_name(self, text):
        self.track_name = text.strip() or 'Track 1'

    def _cb_width(self, val):
        self.width = float(val)
        self._redraw()

    def _toggle_snap(self, _=None):
        self._snap_grid = not self._snap_grid
        lbl = f'SNAP {"ON " if self._snap_grid else "OFF"}'
        self.btn_snap.label.set_text(lbl)
        self.btn_snap.label.set_color(PAL['green'] if self._snap_grid else PAL['text_secondary'])
        self.fig.canvas.draw_idle()

    def _toggle_labels(self, _=None):
        self._show_cp_labels = not self._show_cp_labels
        lbl = f'LABELS {"ON " if self._show_cp_labels else "OFF"}'
        self.btn_label.label.set_text(lbl)
        self.btn_label.label.set_color(PAL['cyan'] if self._show_cp_labels else PAL['text_secondary'])
        self._redraw()

    def _cb_key(self, event):
        if event.key in ('ctrl+z', 'cmd+z'):
            self._undo()
        elif event.key == 'ctrl+s':
            self._save_track_json()
        elif event.key == 'delete' or event.key == 'backspace':
            if self.points:
                self._push_undo()
                self.points.pop()
                self._redraw()
        # Clipboard in TextBox
        if self.txt_name.active and tk:
            try:
                root = tk.Tk(); root.withdraw()
                if event.key == 'ctrl+c':
                    root.clipboard_clear()
                    root.clipboard_append(self.txt_name.text)
                    root.update()
                elif event.key == 'ctrl+v':
                    pasted = root.clipboard_get()
                    self.txt_name.set_val(self.txt_name.text + pasted)
                root.destroy()
            except Exception:
                pass

    def _snap(self, x, y):
        if not self._snap_grid:
            return x, y
        g = self._grid_size
        return round(x / g) * g, round(y / g) * g

    def _nearest_point(self, x, y):
        if not self.points:
            return None
        pts = np.array(self.points)
        d2  = (pts[:, 0] - x)**2 + (pts[:, 1] - y)**2
        idx = int(np.argmin(d2))
        return idx if np.sqrt(d2[idx]) <= SNAP_R else None

    def _cb_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = self._snap(event.xdata, event.ydata)

        if event.button == 1:
            idx = self._nearest_point(x, y)
            if idx is not None:
                self._drag_idx = idx
            else:
                self._push_undo()
                self.points.append([x, y])
                self._redraw()

        elif event.button == 3:
            idx = self._nearest_point(x, y)
            if idx is not None:
                self._push_undo()
                self.points.pop(idx)
                self._redraw()

    def _cb_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = self._snap(event.xdata, event.ydata)

        if self._drag_idx is not None:
            self.points[self._drag_idx] = [x, y]
            self._redraw()
            return

        # Hover highlight
        idx = self._nearest_point(x, y)
        if idx != self._hovered_idx:
            self._hovered_idx = idx
            self._redraw()

    def _cb_release(self, _):
        self._drag_idx = None

    # ── DRAWING ──────────────────────────────────
    def _redraw(self):
        ax = self.ax
        ax.cla()
        self._config_canvas()

        n = len(self.points)

        if n >= 1:
            pts_np = np.array(self.points)
            pad = max(self.width * 3.5, 2.5)
            ax.set_xlim(pts_np[:, 0].min() - pad, pts_np[:, 0].max() + pad)
            ax.set_ylim(pts_np[:, 1].min() - pad, pts_np[:, 1].max() + pad)

        if n < 3:
            self._draw_placeholder(n)
            self.minimap.update()
            self.stats_panel.update(n_pts=n)
            self.curv_chart.update()
            self.fig.canvas.draw_idle()
            return

        pts_arr = np.array(self.points)
        cl = catmull_rom(pts_arr)
        hw = self.width / 2.0

        self._draw_track_surface(ax, cl, hw)
        self._draw_start_finish(ax, cl, pts_arr, hw)
        self._draw_control_points(ax, pts_arr)
        self._draw_info_overlay(ax, cl, hw, n)

        # Right panels
        self.minimap.update(cl, hw, self.points)
        self.stats_panel.update(cl, hw, n, self.track_name)
        self.curv_chart.update(cl)

        self.fig.canvas.draw_idle()

    def _draw_placeholder(self, n):
        ax = self.ax
        if self.points:
            pts = np.array(self.points)
            ax.plot(pts[:, 0], pts[:, 1], 'o--',
                    color=PAL['cp_color'], markersize=10, linewidth=1.5, alpha=0.6)
            for i, (px, py) in enumerate(self.points, 1):
                ax.text(px + 0.15, py + 0.2, str(i),
                        color=PAL['cp_color'], fontsize=8, fontweight='bold',
                        fontfamily='monospace')
        cx = sum(ax.get_xlim()) / 2
        cy = sum(ax.get_ylim()) / 2
        ax.text(cx, cy, f'Add at least 3 points  ({n}/3)',
                ha='center', va='center', fontsize=13,
                color=PAL['white'], alpha=0.35, fontfamily='monospace')

    def _draw_track_surface(self, ax, cl, hw):
        # ── Outer glow ────────────────────────────
        N = len(cl)
        ox, oy, ix, iy = track_borders(cl, hw)

        # Glow band (slightly wider, blurred via alpha stacking)
        for glow_extra, alpha in [(hw * 0.35, 0.07), (hw * 0.2, 0.10), (hw * 0.08, 0.12)]:
            gox, goy, gix, giy = track_borders(cl, hw + glow_extra)
            verts_g = np.concatenate([
                np.column_stack([gox, goy]),
                np.column_stack([gox[[0]], goy[[0]]]),
                np.column_stack([gix[::-1], giy[::-1]]),
                np.column_stack([gix[[0]], giy[[0]]]),
            ])
            codes_g = (
                [MplPath.MOVETO] + [MplPath.LINETO]*(N-1) + [MplPath.CLOSEPOLY] +
                [MplPath.MOVETO] + [MplPath.LINETO]*(N-1) + [MplPath.CLOSEPOLY]
            )
            ax.add_patch(PathPatch(
                MplPath(verts_g, codes_g),
                facecolor=PAL['cyan'], edgecolor='none', zorder=1, alpha=alpha
            ))

        # ── Asphalt fill ──────────────────────────
        outer_v = np.column_stack([ox, oy])
        inner_v = np.column_stack([ix[::-1], iy[::-1]])
        verts = np.concatenate([outer_v, outer_v[0:1], inner_v, inner_v[0:1]])
        codes = np.array(
            [MplPath.MOVETO] + [MplPath.LINETO]*(N-1) + [MplPath.CLOSEPOLY] +
            [MplPath.MOVETO] + [MplPath.LINETO]*(N-1) + [MplPath.CLOSEPOLY],
            dtype=np.uint8)
        ax.add_patch(PathPatch(
            MplPath(verts, codes),
            facecolor=PAL['asphalt'], edgecolor='none', zorder=2, alpha=0.95
        ))

        # ── Curvature heatmap overlay on centerline ──
        if self._show_curv_heat:
            kap = np.abs(curvature(cl))
            cx_arr = np.append(cl[:, 0], cl[0, 0])
            cy_arr = np.append(cl[:, 1], cl[0, 1])
            kv     = np.append(kap, kap[0])
            # Colour map: green (straight) → yellow → red (tight)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'track', ['#7BFF00','#c9a84c','#FD2C2C'])
            lc = gradient_segments(cx_arr, cy_arr, kv, cmap, 0, kap.max()+1e-9)
            lc.set_linewidth(max(1.2, hw * 0.6))
            lc.set_alpha(0.55)
            lc.set_zorder(4)
            ax.add_collection(lc)

        # ── Dashed centerline ─────────────────────
        ax.plot(np.append(cl[:, 0], cl[0, 0]),
                np.append(cl[:, 1], cl[0, 1]),
                '--', color=PAL['yellow'], lw=0.9, alpha=0.35, zorder=5, dashes=(8, 10))

        # ── Track borders ─────────────────────────
        for bx, by, col, lw in [
            (ox, oy, PAL['white'], 2.2),
            (ix, iy, PAL['white'], 2.2),
        ]:
            ax.plot(np.append(bx, bx[0]), np.append(by, by[0]),
                    color=col, lw=lw, zorder=6, solid_capstyle='round')

        # Thin accent lines just inside borders (racing kerb feel)
        for bx, by, col in [(ox, oy, PAL['cyan']), (ix, iy, PAL['red'])]:
            ax.plot(np.append(bx, bx[0]), np.append(by, by[0]),
                    color=col, lw=0.6, alpha=0.45, zorder=7)

    def _draw_start_finish(self, ax, cl, pts_arr, hw):
        # Find centerline point nearest midpoint of first two control points
        mid_raw = (pts_arr[0] + pts_arr[1]) / 2.0
        dists   = np.hypot(cl[:, 0] - mid_raw[0], cl[:, 1] - mid_raw[1])
        idx_cl  = int(np.argmin(dists))
        prev_cl = cl[(idx_cl - 1) % len(cl)]
        next_cl = cl[(idx_cl + 1) % len(cl)]
        tang    = next_cl - prev_cl
        cl_mid  = cl[idx_cl]

        ang_car  = float(np.degrees(np.arctan2(tang[1], tang[0])))
        ang_line = ang_car + 90.0

        # Chequered flag strip
        n_squares = 16
        w_sq = (2 * hw) / n_squares
        h_sq = w_sq * 1.3
        x0   = cl_mid[0] - hw
        y0   = cl_mid[1] - h_sq / 2
        t    = Affine2D().rotate_deg_around(cl_mid[0], cl_mid[1], ang_line) + ax.transData
        for i in range(n_squares):
            col = PAL['white'] if i % 2 == 0 else PAL['bg']
            r = Rectangle((x0 + i * w_sq, y0), w_sq, h_sq,
                           facecolor=col, edgecolor='none', zorder=8)
            r.set_transform(t)
            ax.add_patch(r)

        # Finish line border
        tang_norm = tang / (np.hypot(*tang) or 1.0)
        perp      = np.array([-tang_norm[1], tang_norm[0]])
        p1 = cl_mid + perp * hw
        p2 = cl_mid - perp * hw
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=PAL['yellow'], lw=2.2, zorder=9,
                solid_capstyle='round')

        # Direction arrow (on track surface)
        arrow_tip = cl_mid + tang_norm * hw * 0.7
        ax.annotate('',
                    xy=arrow_tip, xytext=cl_mid,
                    arrowprops=dict(arrowstyle='->', color=PAL['red'],
                                    lw=2.2, mutation_scale=16),
                    zorder=10)

        # "START / FINISH" label
        label_pos = cl_mid + perp * (hw + 0.4)
        ax.text(label_pos[0], label_pos[1], 'S/F',
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=PAL['yellow'], fontfamily='monospace', zorder=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=PAL['bg'],
                          edgecolor=PAL['yellow'], alpha=0.75))

    def _draw_control_points(self, ax, pts_arr):
        for i, (px, py) in enumerate(self.points):
            is_first   = (i == 0)
            is_hovered = (i == self._hovered_idx)
            is_dragged = (i == self._drag_idx)

            # Outer glow ring
            ring_r = 0.32 if is_hovered or is_dragged else 0.22
            ring_col = (PAL['white'] if is_dragged else
                        PAL['cyan']  if is_hovered else
                        PAL['cp_color'])
            ring_alpha = 0.7 if (is_hovered or is_dragged) else 0.4
            ring = Circle((px, py), ring_r,
                           facecolor='none', edgecolor=ring_col,
                           linewidth=1.6, alpha=ring_alpha, zorder=12)
            ax.add_patch(ring)

            # Dot
            dot_color = (PAL['white']    if is_dragged else
                         PAL['green']    if is_first   else
                         PAL['cp_color'])
            dot_size = 80 if is_first else 55
            ax.scatter([px], [py], s=dot_size,
                       color=dot_color, edgecolors=PAL['bg'],
                       linewidths=1.4, zorder=13,
                       path_effects=[pe.withStroke(linewidth=2.5,
                                                   foreground=PAL['bg'])])

            # Label
            if self._show_cp_labels:
                lbl = f'P{i+1}' if not is_first else 'START'
                ax.text(px + 0.18, py + 0.28, lbl,
                        color=dot_color, fontsize=7.5, fontweight='bold',
                        fontfamily='monospace', zorder=14,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground=PAL['bg'])])

        # Thin dashed polygon connecting control points
        if len(self.points) >= 2:
            poly = np.vstack([pts_arr, pts_arr[0]])
            ax.plot(poly[:, 0], poly[:, 1],
                    color=PAL['cp_color'], lw=0.8, ls=':', alpha=0.3, zorder=11)

    def _draw_info_overlay(self, ax, cl, hw, n):
        length = track_length(cl)
        kap    = np.abs(curvature(cl))
        diff   = min(10.0, kap.mean() * 1200 + kap.max() * 400)
        diff_c = PAL['green'] if diff < 4 else PAL['yellow'] if diff < 7 else PAL['red']

        # Bottom-left info pill
        info = (f'  {n} pts  |  w {self.width:.2f}  |  '
                f'len {length:.1f}  |  diff {diff:.1f}/10  ')
        ax.text(0.012, 0.012, info,
                transform=ax.transAxes, fontsize=8,
                color=PAL['text_primary'], fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor=PAL['bg'], edgecolor=PAL['border'],
                          alpha=0.85), zorder=15)

        # Top-left track name badge
        ax.text(0.012, 0.983, f'  {self.track_name.upper()}  ',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=PAL['white'], va='top', ha='left',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor=PAL['inactive'],
                          edgecolor=PAL['border'], alpha=0.9), zorder=15)

        # Top-right undo-stack indicator
        if self._undo_stack:
            ax.text(0.988, 0.983, f'↩ {len(self._undo_stack)}',
                    transform=ax.transAxes, fontsize=7.5,
                    color=PAL['text_secondary'], va='top', ha='right',
                    fontfamily='monospace', zorder=15)

        # Snap indicator
        if self._snap_grid:
            ax.text(0.988, 0.952, f'SNAP {self._grid_size:.1f}',
                    transform=ax.transAxes, fontsize=7,
                    color=PAL['green'], va='top', ha='right',
                    fontfamily='monospace', zorder=15)


# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════╗
║         NEURAL CIRCUIT — Track Editor        ║
╠══════════════════════════════════════════════╣
║  Left-click      add point / drag            ║
║  Right-click     remove nearest point        ║
║  Ctrl+Z / ↩      undo last action            ║
║  Ctrl+S          save track                  ║
║  Delete          remove last point           ║
║  SNAP toggle     snap points to grid         ║
╚══════════════════════════════════════════════╝
""")
    TrackEditor()