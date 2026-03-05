import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mc
import matplotlib.patheffects as _pe
from matplotlib.patches import Rectangle, Polygon, PathPatch, FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import MaxNLocator

from .track import (
    CONFIG, _COR, _CENTERLINE, _CL_X, _CL_Y,
    _TRACK_XLIM, _TRACK_YLIM, _NOME_PISTA,
    gerar_contornos_pista,
)
from .simulation import SimulatorBase
from .neural_network import _N_IN, _N_HID, _N_OUT


# ─────────────────────────────────────────────────────────────
#  COLOUR HELPERS
# ─────────────────────────────────────────────────────────────
def _hex(h):
    return _mc.to_rgba(h)

def _lerp_color(c1, c2, t):
    a, b = _mc.to_rgba(c1), _mc.to_rgba(c2)
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(4))

def _alpha(color, a):
    r = list(_mc.to_rgba(color)); r[3] = a; return tuple(r)


# ─────────────────────────────────────────────────────────────
#  DESIGN TOKENS  (extend _COR with richer palette)
# ─────────────────────────────────────────────────────────────
_UI = {
    'accent':       _COR.get('green',              '#7BFF00'),
    'accent2':      _COR.get('yellow',             '#c9a84c'),
    'danger':       _COR.get('red',                '#FD2C2C'),
    'info':         _COR.get('normal_color',        '#488E97'),
    'best':         _COR.get('best_color',          '#7BFF00'),
    'bg':           _COR.get('black',               '#0a0d12'),
    'card':         _COR.get('card_color',          '#272B30'),
    'card2':                                         '#1e2328',
    'border':                                        '#1e2d42',
    'border_hi':                                     '#2e4a6a',
    'white':        _COR.get('white',               '#e8ecf1'),
    'text1':        _COR.get('primary_text',        '#dce3ea'),
    'text2':        _COR.get('secondary_text',      '#8fa0b0'),
    'text3':        _COR.get('tertiary_text',       '#4a5a70'),
    'inactive':     _COR.get('inactive_gray',       '#2e3a47'),
    'mid':          _COR.get('medium_gray',         '#5a6a7a'),
    'asphalt':      _COR.get('asphalt_color',       '#243040'),
    'grass':        _COR.get('grass_color',         '#0d1520'),
    'dead':         _COR.get('dead_scatter_color',  '#FD2C2C'),
    'normal_car':   _COR.get('normal_color',        '#488E97'),
    'cyan':                                     '#00d4e8',
    'purple':                                   '#7c4dff',
}

_SCALE = 1.0

def _fs(base):
    return max(4.5, base * _SCALE)


# ─────────────────────────────────────────────────────────────
#  LOW-LEVEL DRAW PRIMITIVES  (all blit-safe)
# ─────────────────────────────────────────────────────────────
def _draw_and_remove(ax, artist):
    ax.add_artist(artist)
    ax.draw_artist(artist)
    artist.remove()


def _text(ax, x, y, s, *, fs=8, fw='normal', color=None, ha='left', va='top',
          transform=None, zorder=5, alpha=1.0, glow=False, family='monospace'):
    color = color or _UI['text1']
    transform = transform or ax.transAxes
    pe_list = []
    if glow:
        pe_list = [_pe.withStroke(linewidth=3, foreground=_alpha(color, 0.25))]
    t = ax.text(x, y, s, transform=transform, fontsize=_fs(fs), fontweight=fw,
                color=color, ha=ha, va=va, zorder=zorder, alpha=alpha,
                fontfamily=family, path_effects=pe_list if glow else None)
    ax.draw_artist(t); t.remove()


def _hline(ax, y, xmin=0.0, xmax=1.0, color=None, lw=0.5, alpha=0.4, transform=None):
    color = color or _UI['border']
    transform = transform or ax.transAxes
    l = Line2D([xmin, xmax], [y, y], color=color, lw=lw, alpha=alpha,
               transform=transform, zorder=3)
    ax.add_line(l); ax.draw_artist(l); l.remove()


def _vline(ax, x, ymin=0.0, ymax=1.0, color=None, lw=0.5, alpha=0.4, transform=None):
    color = color or _UI['border']
    transform = transform or ax.transAxes
    l = Line2D([x, x], [ymin, ymax], color=color, lw=lw, alpha=alpha,
               transform=transform, zorder=3)
    ax.add_line(l); ax.draw_artist(l); l.remove()


def _rect(ax, xy, w, h, fc=None, ec='none', lw=0.8, alpha=1.0,
          transform=None, zorder=4, radius=None):
    fc = fc or _UI['card']
    transform = transform or ax.transAxes
    if radius:
        p = FancyBboxPatch(xy, w, h,
                           boxstyle=f'round,pad={radius}',
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           alpha=alpha, transform=transform, zorder=zorder,
                           clip_on=False)
    else:
        p = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=lw,
                      alpha=alpha, transform=transform, zorder=zorder, clip_on=False)
    ax.add_patch(p); ax.draw_artist(p); p.remove()


def _pill(ax, x, y, w, h, *, fc=None, ec=None, lw=1.0, alpha=0.9,
          transform=None, zorder=10):
    fc = fc or _UI['card']
    ec = ec or _UI['border']
    transform = transform or ax.transAxes
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.004',
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, transform=transform, zorder=zorder,
                       clip_on=False)
    ax.add_patch(p); ax.draw_artist(p); p.remove()


def _accent_bar(ax, x, y, h, color=None, transform=None, zorder=11):

    color = color or _UI['accent']
    _rect(ax, (x, y), 0.004, h, fc=color, alpha=0.8,
          transform=transform or ax.transAxes, zorder=zorder)


def _section_header(ax, x, y, label, color=None):
    color = color or _UI['text3']
    _text(ax, x, y, label.upper(), fs=6.5, fw='bold', color=color,
          ha='left', va='top')
    _hline(ax, y - 0.025, xmin=x, xmax=x + 0.9, color=_UI['border'], lw=0.6, alpha=0.6)


# ─────────────────────────────────────────────────────────────
#  GAUGE WIDGET
# ─────────────────────────────────────────────────────────────
def _draw_gauge(ax, cx, cy, r, value, *, color=None, bg_color=None,
                label='', sublabel='', transform=None, zorder=6):
    
    color    = color    or _UI['accent']
    bg_color = bg_color or _UI['inactive']
    transform = transform or ax.transAxes
    start, end = 210, -30          
    sweep = (end - start) * value  

    bg_arc = Arc((cx, cy), 2*r, 2*r, angle=0,
                 theta1=end, theta2=start,
                 color=bg_color, lw=4, transform=transform, zorder=zorder)
    ax.add_patch(bg_arc); ax.draw_artist(bg_arc); bg_arc.remove()

    if value > 0.01:
        fg_arc = Arc((cx, cy), 2*r, 2*r, angle=0,
                     theta1=start + sweep, theta2=start,
                     color=color, lw=4, transform=transform, zorder=zorder + 1)
        ax.add_patch(fg_arc); ax.draw_artist(fg_arc); fg_arc.remove()

    _text(ax, cx, cy + 0.015, f'{value*100:.0f}%',
          fs=7.5, fw='bold', color=color, ha='center', va='center',
          transform=transform, zorder=zorder + 2)
    if label:
        _text(ax, cx, cy - 0.035, label,
              fs=5.5, fw='normal', color=_UI['text3'], ha='center', va='center',
              transform=transform, zorder=zorder + 2)


# ─────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────
class LearningSimulator(SimulatorBase):

    def __init__(self):
        super().__init__()

  
        self._bg                 = None
        self._bg_refresh_pending = False
        self._frame_count        = 0
        self._notifications      = []
        self._best_idx_prev      = None
        self._help_visible       = False
        self._action_log         = []
        self._MAX_LOG            = 13
        self._current_action     = np.array([0.0, 0.5])
        self._fps_samples        = []
        self._last_frame_time    = time.time()
        self._generation_start   = time.time()
        self._ghost_trail        = []   
        self._MAX_TRAIL          = 80
        self._show_trail         = True

        self._build_figure()
        self._precompute_track()
        self.iniciar_tentativa()

    # ─────────────────────────────────────────────────────────
    #  FIGURE LAYOUT
    # ─────────────────────────────────────────────────────────
    def _build_figure(self):
        plt.rcParams.update({
            'figure.facecolor':  _UI['bg'],
            'axes.facecolor':    _UI['card'],
            'axes.edgecolor':    _UI['border'],
            'axes.labelcolor':   _UI['text2'],
            'xtick.color':       _UI['text3'],
            'ytick.color':       _UI['text3'],
            'text.color':        _UI['text1'],
            'font.family':       'monospace',
        })

       
        try:
            import tkinter as _tk
            _r = _tk.Tk(); _r.withdraw()
            SCR_W = _r.winfo_screenwidth()
            SCR_H = _r.winfo_screenheight()
            _r.destroy()
        except Exception:
            SCR_W, SCR_H = 1920, 1080

        DPI = 96
        FW  = SCR_W / DPI         
        FH  = SCR_H / DPI          
        self._dpi = DPI
        self._fig_px_w = SCR_W
        self._fig_px_h = SCR_H

        self.fig = plt.figure(figsize=(FW, FH), facecolor=_UI['bg'], dpi=DPI)

       
        W, H = SCR_W, SCR_H
        def f(x, y, w, h):   
            return [x/W, y/H, w/W, h/H]

        GAP = max(4, int(W * 0.003))   
        PAD = max(4, int(W * 0.003))

        
        BOT_H_PX = max(200, int(H * 0.265))

        
        SB_W_PX  = max(280, int(W * 0.200))

        
        track_x = PAD
        track_y = BOT_H_PX + GAP
        track_w = W - SB_W_PX - GAP - PAD * 2
        track_h = H - track_y - PAD
        self.ax_track = self.fig.add_axes(f(track_x, track_y, track_w, track_h))

   
        sb_x     = W - SB_W_PX - PAD
        lap_h    = max(220, int(H * 0.290))
        radar_h  = max(160, int(H * 0.195))
        pop_h    = H - PAD - lap_h - GAP - radar_h - GAP - BOT_H_PX - GAP - PAD

        lap_y    = H - PAD - lap_h
        radar_y  = lap_y - GAP - radar_h
        pop_y    = BOT_H_PX + GAP

        self.ax_lap   = self.fig.add_axes(f(sb_x, lap_y,   SB_W_PX, lap_h))
        self.ax_radar = self.fig.add_axes(f(sb_x, radar_y, SB_W_PX, radar_h))
        self.ax_mini  = self.fig.add_axes(f(sb_x, pop_y,   SB_W_PX, max(10, pop_h)))

        
        bot_w_total = W - PAD * 2
        wr = [1.70, 0.72, 0.63, 0.80, 1.00]
        tot = sum(wr)
        n_gaps = len(wr) - 1
        usable = bot_w_total - GAP * n_gaps
        cx = PAD
        axes_bot = []
        for w in wr:
            pw = int(usable * w / tot)
            axes_bot.append(self.fig.add_axes(
                f(cx, PAD, pw, BOT_H_PX - PAD)))
            cx += pw + GAP

        self.ax_graph, self.ax_info, self.ax_commands, self.ax_events, self.ax_neural = axes_bot

        
        self._scale = min(W / 1920, H / 1080)   # 1.0 on 1920×1080
        global _SCALE
        _SCALE = self._scale

        # Style all card axes
        for ax in [self.ax_track, self.ax_graph, self.ax_info, self.ax_commands,
                   self.ax_events, self.ax_neural, self.ax_lap, self.ax_radar, self.ax_mini]:
            ax.set_facecolor(_UI['card'])
            for sp in ax.spines.values():
                sp.set_edgecolor(_UI['border']); sp.set_linewidth(0.8)
            ax.tick_params(colors=_UI['text3'], labelsize=max(5, int(6.5 * self._scale)))

        self.fig.canvas.mpl_connect('key_press_event',      self._on_key)
        self.fig.canvas.mpl_connect('close_event',          self._on_close)
        self.fig.canvas.mpl_connect('button_press_event',   self._on_click)


    # ─────────────────────────────────────────────────────────
    #  TRACK GEOMETRY  (computed once)
    # ─────────────────────────────────────────────────────────
    def _precompute_track(self):
        ext_x, ext_y, int_x, int_y = gerar_contornos_pista()
        N, M = len(ext_x), len(int_x)
        outer = np.column_stack([ext_x, ext_y])
        inner = np.column_stack([int_x[::-1], int_y[::-1]])
        verts = np.concatenate([outer, outer[0:1], inner, inner[0:1]])
        codes = np.array(
            [MplPath.MOVETO] + [MplPath.LINETO] * (N - 1) + [MplPath.CLOSEPOLY] +
            [MplPath.MOVETO] + [MplPath.LINETO] * (M - 1) + [MplPath.CLOSEPOLY],
            dtype=np.uint8)
        self._track_path = MplPath(verts, codes)
        self._ext_x = np.append(ext_x, ext_x[0])
        self._ext_y = np.append(ext_y, ext_y[0])
        self._int_x = np.append(int_x, int_x[0])
        self._int_y = np.append(int_y, int_y[0])
        self._cl_cx = np.append(_CENTERLINE[:, 0], _CENTERLINE[0, 0])
        self._cl_cy = np.append(_CENTERLINE[:, 1], _CENTERLINE[0, 1])
        self._start_zebras = self._build_zebras(
            CONFIG['start_line'], (_UI['white'], _UI['bg']))


    def _build_zebras(self, cfg, colors):
        xc, yc = cfg['x'], cfg['y']
        w  = cfg.get('width',       cfg.get('largura',    1.8))
        h  = cfg.get('height',      cfg.get('altura',     0.4))
        ang = cfg.get('angle',      cfg.get('angulo',     0.0))
        n   = cfg.get('num_stripes',cfg.get('num_zebras', 14))
        wz  = w / n; x0 = xc - w / 2
        return [dict(xy=(x0 + i*wz, yc - h/2), width=wz, height=h,
                     cor=colors[i % 2], cx=xc, cy=yc, angle=ang)
                for i in range(n)]


    # ─────────────────────────────────────────────────────────
    #  STATIC BACKGROUND  (saved for blit)
    # ─────────────────────────────────────────────────────────
    def _prepare_background(self):
        self._draw_track_canvas()
        self._draw_bottom_panels()
        self._draw_sidebar_static()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)


    # ─────────────────────────────────────────────────────────
    #  TRACK CANVAS (static part)
    # ─────────────────────────────────────────────────────────
    def _draw_track_canvas(self):
        ax = self.ax_track
        ax.clear()
        ax.set_xlim(*_TRACK_XLIM); ax.set_ylim(*_TRACK_YLIM)
        ax.set_facecolor(_UI['grass'])
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(_UI['border_hi']); sp.set_linewidth(1.2)


        ax.grid(True, color=_UI['border'], alpha=0.06, linewidth=0.4, linestyle=':')


        ax.add_patch(PathPatch(self._track_path,
                               facecolor=_UI['asphalt'], edgecolor='none', zorder=1))


        for extra, a in [(1.8, 0.04), (1.0, 0.06), (0.4, 0.09)]:
            glow_patch = PathPatch(self._track_path,
                                   facecolor='none',
                                   edgecolor=_UI['cyan'],
                                   linewidth=extra * 4,
                                   alpha=a, zorder=2)
            ax.add_patch(glow_patch)


        for bx, by, col, lw in [
            (self._ext_x, self._ext_y, _UI['white'], 1.8),
            (self._int_x, self._int_y, _UI['white'], 1.8),
        ]:
            ax.plot(bx, by, color=col, lw=lw, zorder=4, solid_capstyle='round')


        ax.plot(self._ext_x, self._ext_y, color=_UI['cyan'], lw=0.5, alpha=0.4, zorder=5)
        ax.plot(self._int_x, self._int_y, color=_UI['danger'], lw=0.5, alpha=0.35, zorder=5)

   
        ax.plot(self._cl_cx, self._cl_cy,
                linestyle=(0, (6, 9)), color=_UI['text3'],
                lw=0.7, alpha=0.4, zorder=3)


        for z in self._start_zebras:
            t_xf = Affine2D().rotate_deg_around(z['cx'], z['cy'], z['angle']) + ax.transData
            p = Rectangle(z['xy'], z['width'], z['height'],
                          facecolor=z['cor'], edgecolor='none', zorder=6)
            p.set_transform(t_xf); ax.add_patch(p)


        ax.text(0.5, 0.980, _NOME_PISTA.upper(),
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color=_UI['text1'], va='top', ha='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor=_alpha(_UI['card'], 0.85),
                          edgecolor=_UI['border_hi'], linewidth=0.9),
                zorder=10)


        for (bx, by, bw, bh) in [(0.005, 0.012, 0.035, 0.055),
                                  (0.960, 0.012, 0.035, 0.055)]:
            for dx, dy, sides in [
                (bx,      by+bh, ['top', 'left']),
                (bx+bw,   by+bh, ['top', 'right']),
                (bx,      by,    ['bottom', 'left']),
                (bx+bw,   by,    ['bottom', 'right']),
            ]:
                for side in sides:
                    if side == 'top':
                        ax.plot([dx, dx + (0.008 if 'left' in sides else -0.008)],
                                [dy, dy], color=_UI['accent'], lw=1.2, alpha=0.5,
                                transform=ax.transAxes, zorder=11)
                    elif side == 'bottom':
                        ax.plot([dx, dx + (0.008 if 'left' in sides else -0.008)],
                                [dy, dy], color=_UI['accent'], lw=1.2, alpha=0.5,
                                transform=ax.transAxes, zorder=11)
                    elif side == 'left':
                        ax.plot([dx, dx],
                                [dy, dy + (-0.025 if 'top' in sides else 0.025)],
                                color=_UI['accent'], lw=1.2, alpha=0.5,
                                transform=ax.transAxes, zorder=11)
                    elif side == 'right':
                        ax.plot([dx, dx],
                                [dy, dy + (-0.025 if 'top' in sides else 0.025)],
                                color=_UI['accent'], lw=1.2, alpha=0.5,
                                transform=ax.transAxes, zorder=11)

    # ─────────────────────────────────────────────────────────
    #  BOTTOM PANELS  (static — called once per generation)
    # ─────────────────────────────────────────────────────────
    def _draw_bottom_panels(self):
        self._draw_graph_panel()
        self._draw_info_panel()
        self._clear_commands_panel()
        self._clear_events_panel()
        self._clear_neural_panel()

    def _style_card(self, ax, title=''):
        ax.clear()
        ax.set_facecolor(_UI['card'])
        for sp in ax.spines.values():
            sp.set_edgecolor(_UI['border']); sp.set_linewidth(0.8)
        ax.tick_params(colors=_UI['text3'], labelsize=6.5)
        if title:
            ax.set_title(title, fontsize=8.5, fontweight='bold',
                         color=_UI['text2'], pad=5, fontfamily='monospace')

    def _draw_graph_panel(self):
        ax = self.ax_graph
        self._style_card(ax, 'LEARNING CURVE')
        ax.set_facecolor(_UI['card2'])

        WINDOW = 200
        if not self.historico_melhor:
            ax.text(0.5, 0.5, 'WAITING FOR DATA…', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8,
                    color=_UI['text3'], fontfamily='monospace')
            return

        total  = len(self.historico_melhor)
        start  = max(0, total - WINDOW)
        gens   = np.arange(start, total)
        best   = np.array(self.historico_melhor[start:])
        avg    = np.array(self.historico_media[start:])


        ax.fill_between(gens, best, alpha=0.12, color=_UI['accent'])
        ax.fill_between(gens, avg,  alpha=0.07, color=_UI['info'])


        if len(best) > 5:
            alpha_ema = 0.15
            ema = best.copy()
            for i in range(1, len(ema)):
                ema[i] = alpha_ema * ema[i] + (1 - alpha_ema) * ema[i-1]
            ax.plot(gens, ema, color=_alpha(_UI['accent'], 0.35), lw=1.2,
                    linestyle='--', label='_nolegend_')


        ax.plot(gens, best, color=_UI['accent'], lw=2.0, label='Best',
                solid_capstyle='round')
        ax.plot(gens, avg,  color=_UI['info'],   lw=1.3, label='Avg',
                alpha=0.8,  solid_capstyle='round')


        ax.scatter([gens[-1]], [best[-1]], s=30, color=_UI['accent'],
                   zorder=6, edgecolors=_UI['bg'], linewidths=0.8)
        ax.scatter([gens[-1]], [avg[-1]],  s=18, color=_UI['info'],
                   zorder=6, edgecolors=_UI['bg'], linewidths=0.8)


        ax.set_xlim(start, start + WINDOW - 1)
        ax.set_xlabel('Generation', fontsize=7, color=_UI['text3'])
        ax.grid(True, alpha=0.10, color=_UI['mid'], linestyle=':')
        ax.yaxis.set_major_locator(MaxNLocator(4))


        leg = ax.legend(loc='upper left', fontsize=7, framealpha=0.0)
        for t in leg.get_texts():
            t.set_color(_UI['text2']); t.set_fontfamily('monospace')


    def _draw_info_panel(self):
        ax = self.ax_info
        self._style_card(ax, 'SIMULATION')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_visible(False)

        pop    = CONFIG['simulation']['population']
        laps   = CONFIG['simulation']['target_laps']
        stag   = self.gens_sem_melhora
        mut    = f'{self.taxa_mutacao_atual*100:.0f}%'
        mode   = self.modo_mutacao
        avg_lp = f'{self.avg_laps_gen:.2f}'

        stag_c = (_UI['accent'] if stag < 5 else
                  _UI['accent2'] if stag < 15 else _UI['danger'])
        mut_c  = (_UI['danger']  if mode in ('CRITICAL', 'plateau') else
                  _UI['accent2'] if mode == 'shake' else _UI['text2'])

        sections = [

            (0.96, 'ARCHITECTURE', [
                ('Inputs',     '7 sensors + speed', _UI['text2']),
                ('Layers',     '8 → 16 → 2',        _UI['cyan']),
                ('Activation', 'tanh / sigmoid',     _UI['text2']),
            ]),
            (0.60, 'TRAINING', [
                ('Population', f'{pop} agents', _UI['text2']),
                ('Objective',  f'{laps} laps',  _UI['text2']),
            ]),
            (0.37, 'RUNTIME', [
                ('Stagnation', f'{stag} gen',       stag_c),
                ('Avg laps',   avg_lp,               _UI['text2']),
                ('Mutation',   f'{mut} [{mode}]',    mut_c),
            ]),
        ]


        xL, xR = 0.06, 0.97
        for y0, hdr, rows in sections:
            _text(ax, xL, y0, hdr, fs=6.5, fw='bold', color=_UI['mid'])
            _hline(ax, y0 - 0.035, xmin=xL, xmax=xR,
                   color=_UI['border'], lw=0.6, alpha=0.7, transform=ax.transAxes)
            dy = 0.09
            for i, (lbl, val, col) in enumerate(rows):
                y = y0 - 0.06 - i * dy
                _text(ax, xL, y, lbl, fs=7, color=_UI['text3'], va='center')
                _text(ax, xR, y, str(val), fs=7, fw='bold', color=col,
                      ha='right', va='center')


    def _clear_commands_panel(self):
        ax = self.ax_commands
        self._style_card(ax, 'AI COMMANDS')
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_visible(False)


    def _clear_events_panel(self):
        ax = self.ax_events
        self._style_card(ax, 'EVENTS')
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_visible(False)


    def _clear_neural_panel(self):
        ax = self.ax_neural
        self._style_card(ax, 'NEURAL NETWORK')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_visible(False)


    # ─────────────────────────────────────────────────────────
    #  SIDEBAR  (static)
    # ─────────────────────────────────────────────────────────
    def _draw_sidebar_static(self):
        for ax in (self.ax_lap, self.ax_radar, self.ax_mini):
            self._style_card(ax)
            for sp in ax.spines.values():
                sp.set_edgecolor(_UI['border_hi']); sp.set_linewidth(0.9)


        self.ax_lap.set_xlim(0, 1); self.ax_lap.set_ylim(0, 1)
        self.ax_lap.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)
        for sp in self.ax_lap.spines.values(): sp.set_visible(False)
        self.ax_lap.text(0.5, 0.975, 'BEST ALIVE',
                         transform=self.ax_lap.transAxes,
                         fontsize=8.5, fontweight='bold',
                         color=_UI['text2'], va='top', ha='center',
                         fontfamily='monospace')


        self.ax_radar.set_xlim(-1.25, 1.25); self.ax_radar.set_ylim(-1.25, 1.25)
        self.ax_radar.set_aspect('equal')
        self.ax_radar.tick_params(left=False, bottom=False,
                                  labelleft=False, labelbottom=False)
        for sp in self.ax_radar.spines.values(): sp.set_visible(False)
        self.ax_radar.text(0.5, 0.975, 'SENSOR RADAR',
                           transform=self.ax_radar.transAxes,
                           fontsize=8.5, fontweight='bold',
                           color=_UI['text2'], va='top', ha='center',
                           fontfamily='monospace')


        self.ax_mini.set_facecolor(_UI['card2'])
        self.ax_mini.text(0.5, 0.975, 'POPULATION STATS',
                          transform=self.ax_mini.transAxes,
                          fontsize=8.5, fontweight='bold',
                          color=_UI['text2'], va='top', ha='center',
                          fontfamily='monospace')

    # ─────────────────────────────────────────────────────────
    #  BLIT: NEURAL NETWORK
    # ─────────────────────────────────────────────────────────
    def _draw_neural_network(self, best_idx):
        ax = self.ax_neural
        if best_idx is None:
            return

        car    = self.carrinhos[best_idx]
        brain  = self.cerebros[best_idx]
        s      = np.clip(getattr(car, '_cached_sensores', np.zeros(8)), 0, 1)
        a1     = 1.0 / (1.0 + np.exp(-np.clip(s @ brain.W1 + brain.b1, -500, 500)))
        a2     = 1.0 / (1.0 + np.exp(-np.clip(a1 @ brain.W2 + brain.b2, -500, 500)))

        x_in, x_h, x_out = 0.18, 0.52, 0.78
        in_ys  = np.linspace(0.88, 0.06, 8)
        h_ys   = np.linspace(0.92, 0.04, _N_HID)
        out_ys = np.array([0.68, 0.28])

   
        for x, lbl in [(x_in, 'IN'), (x_h, 'HIDDEN'), (x_out, 'OUT')]:
            _text(ax, x, 0.97, lbl, fs=5.5, fw='bold',
                  color=_UI['mid'], ha='center', va='top')

        c_pos = _hex(_UI['accent'])
        c_neg = _hex(_UI['danger'])


        w1_max = max(np.abs(brain.W1).max(), 1e-6)
        segs, cols, lws = [], [], []
        for i in range(8):
            for j in range(_N_HID):
                w = brain.W1[i, j]; aw = abs(w) / w1_max
                if aw < 0.12: continue
                segs.append([(x_in, in_ys[i]), (x_h, h_ys[j])])
                base = c_pos if w > 0 else c_neg
                cols.append((base[0], base[1], base[2], aw * 0.40))
                lws.append(0.25 + aw * 0.5)
        if segs:
            lc = LineCollection(segs, colors=cols, linewidths=lws,
                                transform=ax.transAxes, zorder=2)
            ax.add_collection(lc); ax.draw_artist(lc); lc.remove()


        w2_max = max(np.abs(brain.W2).max(), 1e-6)
        segs, cols, lws = [], [], []
        for j in range(_N_HID):
            for k in range(_N_OUT):
                w = brain.W2[j, k]; aw = abs(w) / w2_max
                if aw < 0.08: continue
                segs.append([(x_h, h_ys[j]), (x_out, out_ys[k])])
                base = c_pos if w > 0 else c_neg
                cols.append((base[0], base[1], base[2], aw * 0.55))
                lws.append(0.35 + aw * 0.8)
        if segs:
            lc = LineCollection(segs, colors=cols, linewidths=lws,
                                transform=ax.transAxes, zorder=2)
            ax.add_collection(lc); ax.draw_artist(lc); lc.remove()


        c_in = _hex(_UI['info'])
        in_rgba = [(c_in[0], c_in[1], c_in[2], 0.20 + float(v) * 0.80) for v in s]
        sc = ax.scatter([x_in]*8, in_ys, s=26, c=in_rgba, edgecolors='none',
                        transform=ax.transAxes, zorder=5, clip_on=False)
        ax.draw_artist(sc); sc.remove()


        c_on  = _hex(_UI['accent'])
        c_off = _hex(_UI['inactive'])
        h_rgba = []
        for v in a1:
            v = float(v)
            h_rgba.append((c_on[0]*v + c_off[0]*(1-v),
                           c_on[1]*v + c_off[1]*(1-v),
                           c_on[2]*v + c_off[2]*(1-v),
                           0.22 + v * 0.78))
        sc = ax.scatter([x_h]*_N_HID, h_ys, s=14, c=h_rgba, edgecolors='none',
                        transform=ax.transAxes, zorder=5, clip_on=False)
        ax.draw_artist(sc); sc.remove()


        steer_v = float((a2[0] - 0.5) * 2.0)
        gas_v   = float(a2[1])
        c_y = _hex(_UI['accent2'])
        c_g = _hex(_UI['accent'])
        c_r = _hex(_UI['danger'])
        out_colors = [
            (c_y[0], c_y[1], c_y[2], 0.3 + abs(steer_v) * 0.70),
            ((c_g[0] if gas_v >= 0.5 else c_r[0]),
             (c_g[1] if gas_v >= 0.5 else c_r[1]),
             (c_g[2] if gas_v >= 0.5 else c_r[2]),
             0.3 + abs(gas_v - 0.5) * 2 * 0.70),
        ]
        sc = ax.scatter([x_out]*2, out_ys, s=52, c=out_colors, edgecolors='none',
                        transform=ax.transAxes, zorder=5, clip_on=False)
        ax.draw_artist(sc); sc.remove()


        for y, lbl in zip(in_ys, ['-90°','-45°','-22°',' 0° ','+22°','+45°','+90°','vel']):
            _text(ax, 0.01, y, lbl, fs=5.5, color=_UI['text3'],
                  ha='left', va='center')


        esq   = max(0.0, -steer_v); dir_ = max(0.0, steer_v)
        gas_i = max(0.0, (gas_v - 0.5) * 2); brk_i = max(0.0, (0.5 - gas_v) * 2)
        for y_n, lbl, rgba, intensity, dy_ in [
            (out_ys[0], 'LFT', c_y, esq,  0.10),
            (out_ys[0], 'RGT', c_y, dir_, 0.00),
            (out_ys[1], 'GAS', c_g, gas_i, 0.10),
            (out_ys[1], 'BRK', c_r, brk_i, 0.00),
        ]:
            a_ = 0.25 + intensity * 0.75
            fw_ = 'bold' if intensity > 0.25 else 'normal'
            _text(ax, x_out + 0.12, y_n + dy_, lbl,
                  fs=5.5, fw=fw_,
                  color=(rgba[0], rgba[1], rgba[2], a_),
                  ha='left', va='center')

        for y_n, val, rgba in [(out_ys[0], steer_v, c_y),
                               (out_ys[1], gas_v,   c_g if gas_v >= 0.5 else c_r)]:
            _text(ax, x_out + 0.12, y_n - 0.07, f'{val:+.2f}',
                  fs=6.5, fw='bold',
                  color=(rgba[0], rgba[1], rgba[2], 0.9),
                  ha='left', va='center')


    # ─────────────────────────────────────────────────────────
    #  BLIT: SENSOR RADAR
    # ─────────────────────────────────────────────────────────
    def _draw_radar(self, best_idx):
        ax = self.ax_radar
        ax.cla()
        ax.set_facecolor(_UI['card'])
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
        ax.set_aspect('equal')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        _text(ax, 0.5, 0.975, 'SENSOR RADAR',
              fs=8.5, fw='bold', color=_UI['text2'], ha='center', va='top')


        for r in [0.33, 0.66, 1.0]:
            circle = plt.Circle((0, 0), r, fill=False,
                                 edgecolor=_UI['border'], lw=0.6, alpha=0.5,
                                 transform=ax.transData, zorder=1)
            ax.add_patch(circle); ax.draw_artist(circle); circle.remove()

        if best_idx is None:
            return

        car = self.carrinhos[best_idx]
        raw = getattr(car, '_cached_sensores', np.zeros(8))[:7]
        sensor_angs = np.radians([-90, -45, -22.5, 0, 22.5, 45, 90])
        car_rad     = np.radians(car.angulo)

        points = []
        for sa, dist in zip(sensor_angs, raw):
            d_norm = float(dist)  # already 0..1 normalised by sim
            angle  = car_rad + sa
            px = d_norm * np.cos(angle)
            py = d_norm * np.sin(angle)
            points.append((px, py))

        xs = [p[0] for p in points] + [points[0][0]]
        ys = [p[1] for p in points] + [points[0][1]]

        fill = Polygon(list(zip(xs, ys)), closed=True,
                       facecolor=_alpha(_UI['cyan'], 0.12),
                       edgecolor=_UI['cyan'], linewidth=1.0,
                       zorder=3, transform=ax.transData)
        ax.add_patch(fill); ax.draw_artist(fill); fill.remove()

        for px, py, dist in zip(xs[:-1], ys[:-1], raw):
            d_norm = float(dist)
            col = _lerp_color(_UI['danger'], _UI['accent'], d_norm)
            c = plt.Circle((px, py), 0.055, color=col, zorder=5, transform=ax.transData)
            ax.add_patch(c); ax.draw_artist(c); c.remove()

        # Car dot
        cc = plt.Circle((0, 0), 0.08, color=_UI['best'], zorder=6, transform=ax.transData)
        ax.add_patch(cc); ax.draw_artist(cc); cc.remove()

        self.fig.canvas.blit(self.ax_radar.get_tightbbox())


    # ─────────────────────────────────────────────────────────
    #  BLIT: SIDEBAR — BEST ALIVE PANEL
    # ─────────────────────────────────────────────────────────
    def _draw_lap_panel(self, best_idx, alive_count):
        ax = self.ax_lap
        ax.cla()
        ax.set_facecolor(_UI['card'])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp in ax.spines.values():
            sp.set_edgecolor(_UI['border_hi']); sp.set_linewidth(0.9); sp.set_visible(True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


        _rect(ax, (0, 0.875), 1.0, 0.125,
              fc=_alpha(_UI['card2'], 0.9), ec='none', zorder=3)
        _text(ax, 0.5, 0.965, 'BEST ALIVE',
              fs=8.5, fw='bold', color=_UI['accent'], ha='center', va='top', glow=True)


        pop = len(self.carrinhos)
        frac = alive_count / pop
        alive_col = (_UI['accent'] if frac > 0.5 else
                     _UI['accent2'] if frac > 0.2 else _UI['danger'])
        _pill(ax, 0.62, 0.885, 0.35, 0.072,
              fc=_alpha(alive_col, 0.15), ec=alive_col, lw=0.8, zorder=4)
        _text(ax, 0.795, 0.924, f'{alive_count}/{pop}',
              fs=7, fw='bold', color=alive_col, ha='center', va='center')

        gen_str = f'GEN {self.geracao + 1}'
        _pill(ax, 0.03, 0.885, 0.28, 0.072,
              fc=_alpha(_UI['cyan'], 0.12), ec=_UI['cyan'], lw=0.8, zorder=4)
        _text(ax, 0.17, 0.924, gen_str,
              fs=7, fw='bold', color=_UI['cyan'], ha='center', va='center')

        _hline(ax, 0.875, color=_UI['border_hi'], lw=0.8, alpha=0.8)

        if best_idx is None:
            _text(ax, 0.5, 0.5, 'NO ALIVE CARS', fs=8,
                  color=_UI['text3'], ha='center', va='center')
            return

        car = self.carrinhos[best_idx]
        obj = CONFIG['simulation']['target_laps']


        lap_frac = car.voltas_completas / max(obj, 1)
        _draw_gauge(ax, 0.25, 0.62, 0.16,
                    value=min(lap_frac, 1.0),
                    color=_UI['accent'],
                    label=f'{car.voltas_completas}/{obj} LAPS')


        score_max = max(self.historico_melhor[-1] if self.historico_melhor else 1, 1)
        score_frac = min(car.pontos_acumulados / score_max, 1.0)
        _draw_gauge(ax, 0.75, 0.62, 0.16,
                    value=score_frac,
                    color=_UI['accent2'],
                    label='SCORE')

        _text(ax, 0.5, 0.635, f'{car.pontos_acumulados:.0f} pts',
              fs=9, fw='bold', color=_UI['accent2'], ha='center', va='center')

        _hline(ax, 0.440, color=_UI['border'], lw=0.5, alpha=0.5)


        fr_atual  = car.tempo_vivo - car.frame_inicio_volta
        best_lap  = car.melhor_tempo_volta
        record    = self.record_global_volta
        rec_gen   = self.record_global_volta_geracao


        def _row(y_, label, value, col_v):
            _text(ax, 0.06, y_, label, fs=7, color=_UI['text3'], va='center')
            _text(ax, 0.97, y_, value, fs=7.5, fw='bold',
                  color=col_v, ha='right', va='center')


        if best_lap and fr_atual < best_lap:
            cur_col = _UI['accent']
        elif car.voltas_completas > 0:
            cur_col = _UI['accent2']
        else:
            cur_col = _UI['text2']

        _row(0.390, 'Current lap',  f'{fr_atual} fr', cur_col)
        _hline(ax, 0.355, color=_UI['border'], lw=0.4, alpha=0.4)
        _row(0.315, 'Best this run',
             f'{best_lap} fr' if best_lap else '—', _UI['accent'])
        _hline(ax, 0.280, color=_UI['border'], lw=0.4, alpha=0.4)
        _row(0.240, 'Record (all)',
             (f'{record} fr / G{rec_gen}' if record else '—'), _UI['cyan'])
        _hline(ax, 0.205, color=_UI['border'], lw=0.4, alpha=0.4)


        vel_pct = car.velocidade / CONFIG['cars']['max_speed']
        pen = CONFIG['penalties']
        thr = (0.0 if car.voltas_completas == 0 else
               min(0.9, pen['initial_slow_threshold']
                   + (car.voltas_completas - 1) * pen['slow_threshold_increment']))
        spd_col = _UI['danger'] if vel_pct < thr else _UI['accent']

        _text(ax, 0.06, 0.155, 'Speed', fs=7, color=_UI['text3'], va='center')
        _text(ax, 0.97, 0.155, f'{vel_pct*100:.0f}%',
              fs=7.5, fw='bold', color=spd_col, ha='right', va='center')
        _rect(ax, (0.06, 0.085), 0.88, 0.038, fc=_UI['inactive'], ec='none', zorder=3)
        _rect(ax, (0.06, 0.085), 0.88 * vel_pct, 0.038,
              fc=spd_col, ec='none', alpha=0.85, zorder=4)
        if thr > 0:
            _rect(ax, (0.06 + 0.88 * thr - 0.003, 0.080),
                  0.006, 0.048, fc=_UI['danger'], ec='none', alpha=0.7, zorder=5)


        stag      = self.gens_sem_melhora
        stag_max  = 40
        stag_frac = min(stag / stag_max, 1.0)
        stag_col  = (_UI['accent'] if stag < 5 else
                     _UI['accent2'] if stag < 15 else _UI['danger'])
        _text(ax, 0.06, 0.040, f'Stagnation  {stag} gen',
              fs=6.5, color=stag_col, va='center')
        _rect(ax, (0.06, 0.012), 0.88, 0.020, fc=_UI['inactive'], ec='none', zorder=3)
        _rect(ax, (0.06, 0.012), 0.88 * stag_frac, 0.020,
              fc=stag_col, ec='none', alpha=0.75, zorder=4)

        self.fig.canvas.blit(self.ax_lap.get_tightbbox())

    # ─────────────────────────────────────────────────────────
    #  BLIT: SIDEBAR — POPULATION STATS MINI CHART
    # ─────────────────────────────────────────────────────────
    def _draw_mini_panel(self, alive_count):
        ax = self.ax_mini
        ax.cla()
        ax.set_facecolor(_UI['card2'])
        for sp in ax.spines.values():
            sp.set_edgecolor(_UI['border_hi']); sp.set_linewidth(0.9); sp.set_visible(True)
        ax.tick_params(colors=_UI['text3'], labelsize=6)
        _text(ax, 0.5, 0.978, 'POPULATION STATS',
              fs=8.5, fw='bold', color=_UI['text2'], ha='center', va='top')
        _hline(ax, 0.945, color=_UI['border_hi'], lw=0.8, alpha=0.7)


        now = time.time()
        dt  = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps_samples.append(1.0 / dt)
            if len(self._fps_samples) > 30:
                self._fps_samples.pop(0)
        fps = np.mean(self._fps_samples) if self._fps_samples else 0

        pop = len(self.carrinhos)
        dead = pop - alive_count


        frac = alive_count / max(pop, 1)
        _rect(ax, (0.05, 0.870), 0.90, 0.050,
              fc=_UI['inactive'], ec='none', zorder=3)
        _rect(ax, (0.05, 0.870), 0.90 * frac, 0.050,
              fc=_UI['accent'],  ec='none', alpha=0.85, zorder=4)
        _rect(ax, (0.05 + 0.90 * frac, 0.870), 0.90 * (1 - frac), 0.050,
              fc=_UI['danger'], ec='none', alpha=0.35, zorder=4)
        _text(ax, 0.50, 0.897,
              f'{alive_count} alive  /  {dead} dead',
              fs=6.5, color=_UI['text2'], ha='center', va='center')

        _hline(ax, 0.848, color=_UI['border'], lw=0.5, alpha=0.5)


        if len(self.historico_melhor) >= 2:
            WINDOW = 40
            total  = len(self.historico_melhor)
            start  = max(0, total - WINDOW)
            gens   = np.arange(start, total)
            best   = np.array(self.historico_melhor[start:])
            avg    = np.array(self.historico_media[start:])
            ax2    = ax.inset_axes([0.05, 0.490, 0.90, 0.340])
            ax2.set_facecolor(_UI['card2'])
            for sp2 in ax2.spines.values():
                sp2.set_edgecolor(_UI['border']); sp2.set_linewidth(0.5)
            ax2.tick_params(colors=_UI['text3'], labelsize=5)
            ax2.fill_between(gens, best, alpha=0.12, color=_UI['accent'])
            ax2.plot(gens, best, color=_UI['accent'], lw=1.2)
            ax2.plot(gens, avg,  color=_UI['info'],   lw=0.8, alpha=0.7)
            ax2.set_xlim(gens[0], gens[-1])
            ax2.set_xlabel('Gen', fontsize=5, color=_UI['text3'])
            ax2.grid(True, alpha=0.08, color=_UI['mid'], linestyle=':')
            ax2.yaxis.set_major_locator(MaxNLocator(3))
            ax.draw_artist(ax2)

        _hline(ax, 0.460, color=_UI['border'], lw=0.5, alpha=0.5)


        mode = self.modo_mutacao
        mode_colors = {
            'explore':  _UI['info'],
            'normal':   _UI['text2'],
            'fine':     _UI['accent'],
            'shake':    _UI['accent2'],
            'plateau':  _UI['danger'],
            'CRITICAL': _UI['danger'],
        }
        mc = mode_colors.get(mode, _UI['text2'])
        _pill(ax, 0.05, 0.365, 0.90, 0.075,
              fc=_alpha(mc, 0.10), ec=mc, lw=0.8, zorder=3)
        _text(ax, 0.50, 0.405,
              f'MUTATION  {self.taxa_mutacao_atual*100:.0f}%  [{mode}]',
              fs=7, fw='bold', color=mc, ha='center', va='center')

        _hline(ax, 0.345, color=_UI['border'], lw=0.5, alpha=0.5)


        metrics = [
            ('GEN',   f'{self.geracao + 1}',    _UI['cyan']),
            ('POP',   f'{pop}',                  _UI['text2']),
            ('STAG',  f'{self.gens_sem_melhora}', _UI['accent2']),
            ('FPS',   f'{fps:.0f}',              _UI['text3']),
        ]
        mw = 1.0 / len(metrics)
        for i, (lbl, val, col) in enumerate(metrics):
            cx2 = mw * i + mw / 2
            _text(ax, cx2, 0.290, lbl, fs=5.5, color=_UI['text3'],
                  ha='center', va='center')
            _text(ax, cx2, 0.225, val, fs=9, fw='bold', color=col,
                  ha='center', va='center')
            if i < len(metrics) - 1:
                _rect(ax, (mw*(i+1) - 0.003, 0.175), 0.006, 0.135,
                      fc=_UI['border'], ec='none', alpha=0.5, zorder=3)

        _hline(ax, 0.168, color=_UI['border'], lw=0.5, alpha=0.5)


        tips = [
            ('Q',      'quit simulation'),
            ('SPACE',  'toggle sensors'),
        ]
        for i, (k, desc) in enumerate(tips):
            y_ = 0.125 - i * 0.065
            _pill(ax, 0.05, y_ - 0.022, 0.12, 0.042,
                  fc=_UI['inactive'], ec=_UI['mid'], lw=0.6, zorder=3)
            _text(ax, 0.11, y_, k,
                  fs=6, fw='bold', color=_UI['text2'], ha='center', va='center')
            _text(ax, 0.22, y_, desc,
                  fs=6, color=_UI['text3'], ha='left', va='center')

        self.fig.canvas.blit(self.ax_mini.get_tightbbox())


    # ─────────────────────────────────────────────────────────
    #  BLIT: COMMANDS PANEL
    # ─────────────────────────────────────────────────────────
    def _draw_commands_panel(self, best_idx):
        al = self.ax_commands
        if best_idx is None:
            return

        car    = self.carrinhos[best_idx]
        acao   = car.ultima_acao
        steer  = float(acao[0])         
        gas    = float(acao[1])         
        acel   = gas >= 0.5
        vel_pct = car.velocidade / CONFIG['cars']['max_speed']
        pen     = CONFIG['penalties']
        v_laps  = car.voltas_completas
        thr_spd = (0.0 if v_laps == 0 else
                   min(0.9, pen['initial_slow_threshold']
                       + (v_laps - 1) * pen['slow_threshold_increment']))

        acel_c = _UI['accent'] if acel else _UI['danger']
        spd_c  = _UI['danger'] if vel_pct < thr_spd else _UI['accent']


        _text(al, 0.5, 8.85, 'STEERING', fs=6.5, color=_UI['text3'],
              ha='left', va='top', transform=al.transData)

        tx0, tx1, ty, th = 0.5, 9.5, 7.60, 0.52

        _rect(al, (tx0, ty), tx1 - tx0, th,
              fc=_UI['inactive'], ec='none', zorder=3, transform=al.transData)

        cx  = (tx0 + tx1) / 2

        _rect(al, (cx - 0.05, ty - 0.05), 0.10, th + 0.10,
              fc=_UI['mid'], ec='none', alpha=0.6, zorder=4, transform=al.transData)


        nx = cx + steer * (tx1 - cx)
        d_col = (_UI['info']   if abs(steer) < 0.25 else
                 _UI['accent2'] if steer < 0         else _UI['accent'])
        _rect(al, (min(cx, nx), ty), abs(nx - cx), th,
              fc=d_col, ec='none', alpha=0.85, zorder=5, transform=al.transData)

        _rect(al, (nx - 0.10, ty - 0.12), 0.20, th + 0.24,
              fc=_UI['white'], ec='none', zorder=6, transform=al.transData)

        _text(al, 9.5, 8.85, f'{steer:+.2f}', fs=6.5, color=_UI['text2'],
              ha='right', va='top', transform=al.transData)


        for txt, xp, ha_, active in [('◀', tx0 - 0.18, 'right', steer < -0.1),
                                      ('▶', tx1 + 0.18, 'left',  steer > 0.1)]:
            col = _UI['accent'] if active else _UI['inactive']
            _text(al, xp, ty + th/2, txt, fs=6.5, fw='bold', color=col,
                  ha=ha_, va='center', transform=al.transData)


        _text(al, 0.5, 6.10, 'THROTTLE', fs=6.5, color=_UI['text3'],
              ha='left', va='top', transform=al.transData)
        _rect(al, (tx0, 4.92), tx1 - tx0, th,
              fc=_UI['inactive'], ec='none', zorder=3, transform=al.transData)
        _rect(al, (tx0, 4.92), (tx1 - tx0) * gas, th,
              fc=acel_c, ec='none', alpha=0.85, zorder=4, transform=al.transData)
        _text(al, 9.5, 6.10, f'{gas:.2f}', fs=6.5, color=_UI['text2'],
              ha='right', va='top', transform=al.transData)

        for txt, xp, col in [('GAS',   2.5, _UI['accent'] if acel else _UI['text3']),
                              ('BRAKE', 7.5, _UI['danger'] if not acel else _UI['text3'])]:
            fw_ = 'bold' if (acel and txt == 'GAS') or (not acel and txt == 'BRAKE') else 'normal'
            _text(al, xp, 3.55, txt, fs=7.5, fw=fw_, color=col,
                  ha='center', va='center', transform=al.transData)


        l = Line2D([5.0, 5.0], [3.05, 4.10], color=_UI['border'], lw=0.8, zorder=4)
        al.add_line(l); al.draw_artist(l); l.remove()


        _text(al, 0.5, 1.70, 'SPEED', fs=6.5, color=_UI['text3'],
              ha='left', va='top', transform=al.transData)
        _text(al, 9.5, 1.70, f'{vel_pct*100:.0f}%  ({car.velocidade:.2f})',
              fs=6.5, color=_UI['text2'], ha='right', va='top', transform=al.transData)
        _rect(al, (0.5, 0.35), 9.0, 0.65, fc=_UI['inactive'], ec='none',
              zorder=3, transform=al.transData)
        _rect(al, (0.5, 0.35), 9.0 * vel_pct, 0.65, fc=spd_c, ec='none',
              alpha=0.9, zorder=4, transform=al.transData)
        if thr_spd > 0:
            mx = 0.5 + 9.0 * thr_spd
            l2 = Line2D([mx, mx], [0.28, 1.08], color=_UI['danger'],
                        lw=1.2, alpha=0.9, zorder=5)
            al.add_line(l2); al.draw_artist(l2); l2.remove()

    # ─────────────────────────────────────────────────────────
    #  BLIT: EVENTS PANEL
    # ─────────────────────────────────────────────────────────
    def _draw_events_panel(self, best_idx):
        ae = self.ax_events
        if best_idx is None:
            return

        car     = self.carrinhos[best_idx]
        ef      = car.estado_frame
        rec     = CONFIG['rewards']
        pen     = CONFIG['penalties']
        pre_lap = car.voltas_completas == 0
        v       = car.voltas_completas

        peso_dev  = (0.0 if pre_lap else
                     pen['slow_weight_post_lap'] + (v - 1) * pen['aggravator_per_lap'])
        peso_vel  = rec['speed_weight'] + (0 if pre_lap else (v - 1) * pen['aggravator_per_lap'])
        thr       = (0.0 if pre_lap else
                     min(0.9, pen['initial_slow_threshold']
                         + (v - 1) * pen['slow_threshold_increment']))

        rows = [
            ('Speed',     'rapido',    _UI['accent'],
             '—' if pre_lap else f'+{peso_vel:.1f}/fr',
             'locked pre-lap' if pre_lap else f'above {thr*100:.0f}% max spd'),
            ('Lap Done',  'volta',     _UI['accent'],
             f"+{rec['lap_reward']:.0f}", '1 lap bonus'),
            ('Wall',      'parede',    _UI['danger'],
             '—' if pre_lap else f"−{pen['near_wall_penalty']:.0f}",
             'locked pre-lap' if pre_lap else 'near wall'),
            ('Wrong way', 'contramao', _UI['accent2'],
             'FREEZE' if pre_lap else f"−{pen['wrong_way_penalty']:.0f}",
             'wrong direction'),
            ('Too slow',  'devavar',   _UI['accent2'],
             '—' if pre_lap else f'−{peso_dev:.1f}/fr',
             f'below {thr*100:.0f}% (lap {v})' if not pre_lap else 'locked pre-lap'),
        ]

        for k, (label, key, col_on, val_str, descr) in enumerate(rows):
            gated  = pre_lap and key in ('rapido', 'parede', 'devavar')
            active = bool(ef.get(key)) and not gated
            y_row  = 8.55 - k * 1.72

            bar_col = col_on if active else _UI['inactive']
            lbl_col = _UI['text1'] if active else _UI['text3']
            val_col = col_on if active else _UI['mid']
            dsc_col = _UI['text3'] if active else _UI['text3']


            _rect(ae, (0.25, y_row - 0.62), 0.22, 1.32,
                  fc=bar_col, ec='none', alpha=0.85 if active else 0.3,
                  zorder=4, transform=ae.transData)


            if active:
                gp = FancyBboxPatch((0.25, y_row - 0.62), 0.22, 1.32,
                                    boxstyle='round,pad=0.08',
                                    facecolor='none',
                                    edgecolor=col_on, linewidth=1.0,
                                    alpha=0.6, transform=ae.transData, zorder=5)
                ae.add_patch(gp); ae.draw_artist(gp); gp.remove()


            l = Line2D([0.25, 9.75], [y_row - 0.72, y_row - 0.72],
                       color=_UI['border'], lw=0.4, alpha=0.4, zorder=3)
            ae.add_line(l); ae.draw_artist(l); l.remove()


            _text(ae, 0.85, y_row + 0.38, label,
                  fs=7.5, fw='bold' if active else 'normal',
                  color=lbl_col, ha='left', va='center', transform=ae.transData)
            _text(ae, 0.85, y_row - 0.18, descr,
                  fs=5.8, color=dsc_col, ha='left', va='center', transform=ae.transData)
            _text(ae, 9.75, y_row + 0.10, val_str,
                  fs=7.5, fw='bold' if active else 'normal',
                  color=val_col, ha='right', va='center', transform=ae.transData)

    # ─────────────────────────────────────────────────────────
    #  BLIT: CARS ON TRACK
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _car_sprite_verts(cx, cy, angle_deg, length, width):
        """
        Return a list of (x,y) polygon vertices forming a top-down car sprite.
        The car is a rounded rectangle with a windscreen notch and front nose.

        Layout (local coords, car points right):
          rear  ──────────────── front
          |   [wheel]  body  [nose]  |
        """
        L, W = length, width
        rad  = np.radians(angle_deg)
        ca, sa = np.cos(rad), np.sin(rad)

        def _rot(lx, ly):
            return cx + lx*ca - ly*sa, cy + lx*sa + ly*ca


        r = W * 0.18   
        body = [
            _rot(-L*0.45,  W*0.30),   
            _rot(-L*0.45, -W*0.30),   
            _rot( L*0.30, -W*0.48),   
            _rot( L*0.48, -W*0.28),   
            _rot( L*0.55,  0.0),      
            _rot( L*0.48,  W*0.28),   
            _rot( L*0.30,  W*0.48),   
        ]
        return body

    def _draw_cars(self, best_idx, alive_mask):
        ax  = self.ax_track
        cfg = CONFIG['cars']
       
        track_span = min(
            _TRACK_XLIM[1] - _TRACK_XLIM[0],
            _TRACK_YLIM[1] - _TRACK_YLIM[0]
        )
        L = track_span * 0.028   
        W = L * 0.48


        if self._show_trail and self._ghost_trail:
            n = len(self._ghost_trail)
            segs, cols = [], []
            for i in range(1, n):
                segs.append([self._ghost_trail[i-1][:2],
                             self._ghost_trail[i][:2]])
                a = (i / n) * 0.28
                cols.append(_alpha(_UI['accent2'], a))
            if segs:
                lc = LineCollection(segs, colors=cols, linewidths=1.2,
                                    zorder=8, transform=ax.transData)
                ax.add_collection(lc); ax.draw_artist(lc); lc.remove()


        dead_x = [c.x for c, v in zip(self.carrinhos, alive_mask) if not v]
        dead_y = [c.y for c, v in zip(self.carrinhos, alive_mask) if not v]
        if dead_x:
            sc = ax.scatter(dead_x, dead_y, marker='x',
                            c=_UI['dead'], s=cfg['dead_size'],
                            linewidths=cfg['dead_thickness'],
                            alpha=0.20, zorder=9)
            ax.draw_artist(sc); sc.remove()


        body_verts, body_colors = [], []
        wind_verts, wind_colors = [], []

        for i, car in enumerate(self.carrinhos):
            if not car.vivo or i == best_idx:
                continue
            rad = np.radians(car.angulo)
            ca, sa = np.cos(rad), np.sin(rad)
            body = self._car_sprite_verts(car.x, car.y, car.angulo, L, W)
            body_verts.append(body)
            body_colors.append(_alpha(_UI['normal_car'], 0.85))


            wL, wW = L * 0.18, W * 0.38
            wx = car.x + (L*0.15)*ca
            wy = car.y + (L*0.15)*sa
            wverts = [
                (wx - wL*ca + wW*sa, wy - wL*sa - wW*ca),
                (wx - wL*ca - wW*sa, wy - wL*sa + wW*ca),
                (wx + wL*ca - wW*sa, wy + wL*sa + wW*ca),
                (wx + wL*ca + wW*sa, wy + wL*sa - wW*ca),
            ]
            wind_verts.append(wverts)
            wind_colors.append(_alpha(_UI['cyan'], 0.30))

        if body_verts:
            pc = PolyCollection(body_verts, facecolors=body_colors,
                                edgecolors=_alpha(_UI['normal_car'], 0.4),
                                linewidths=0.4, zorder=10)
            ax.add_collection(pc); ax.draw_artist(pc); pc.remove()
        if wind_verts:
            wc = PolyCollection(wind_verts, facecolors=wind_colors,
                                edgecolors='none', zorder=11)
            ax.add_collection(wc); ax.draw_artist(wc); wc.remove()


        if best_idx is not None:
            car = self.carrinhos[best_idx]
            if car.vivo:
                rad = np.radians(car.angulo)
                ca, sa = np.cos(rad), np.sin(rad)

                
                for gL, gW, ga in [(L*2.0, W*2.0, 0.06),
                                   (L*1.5, W*1.5, 0.10),
                                   (L*1.2, W*1.2, 0.14)]:
                    gverts = self._car_sprite_verts(car.x, car.y, car.angulo, gL, gW)
                    gp = Polygon(gverts, closed=True,
                                 facecolor=_UI['best'], edgecolor='none',
                                 alpha=ga, zorder=9)
                    ax.add_patch(gp); ax.draw_artist(gp); gp.remove()


                bverts = self._car_sprite_verts(car.x, car.y, car.angulo, L, W)
                bp = Polygon(bverts, closed=True,
                             facecolor=_UI['best'],
                             edgecolor=_alpha(_UI['white'], 0.6),
                             linewidth=0.6, zorder=12)
                ax.add_patch(bp); ax.draw_artist(bp); bp.remove()


                wL2, wW2 = L * 0.18, W * 0.38
                wx2 = car.x + (L*0.15)*ca
                wy2 = car.y + (L*0.15)*sa
                wv2 = [
                    (wx2 - wL2*ca + wW2*sa, wy2 - wL2*sa - wW2*ca),
                    (wx2 - wL2*ca - wW2*sa, wy2 - wL2*sa + wW2*ca),
                    (wx2 + wL2*ca - wW2*sa, wy2 + wL2*sa + wW2*ca),
                    (wx2 + wL2*ca + wW2*sa, wy2 + wL2*sa - wW2*ca),
                ]
                wp = Polygon(wv2, closed=True,
                             facecolor=_UI['cyan'], edgecolor='none',
                             alpha=0.85, zorder=13)
                ax.add_patch(wp); ax.draw_artist(wp); wp.remove()


                wh_l, wh_w = L * 0.22, W * 0.14
                for lx, ly in [(-L*0.28,  W*0.42), (-L*0.28, -W*0.42),
                                ( L*0.15,  W*0.45), ( L*0.15, -W*0.45)]:
                    wxc = car.x + lx*ca - ly*sa
                    wyc = car.y + lx*sa + ly*ca
                    wh_verts = self._car_sprite_verts(wxc, wyc, car.angulo, wh_l, wh_w)
                    wh_p = Polygon(wh_verts, closed=True,
                                   facecolor=_alpha(_UI['bg'], 0.9),
                                   edgecolor=_alpha(_UI['text3'], 0.5),
                                   linewidth=0.3, zorder=13)
                    ax.add_patch(wh_p); ax.draw_artist(wh_p); wh_p.remove()

   
                if CONFIG.get('visualization', {}).get('show_sensors', True):
                    sensores = getattr(car, '_cached_sensores', None) or car.get_sensores()
                    angs = [car.angulo + a for a in [-90,-45,-22.5,0,22.5,45,90]]
                    sensor_cols = _COR.get('sensor_colors', [_UI['danger'],_UI['accent2'],_UI['accent']])
                    for ang, dist, sc_ in zip(angs, sensores[:7]*15, sensor_cols):
                        r2 = np.radians(ang)
                        sl = Line2D([car.x, car.x + dist*np.cos(r2)],
                                    [car.y, car.y + dist*np.sin(r2)],
                                    ls='--', color=sc_, alpha=0.50, lw=1.0)
                        ax.add_line(sl); ax.draw_artist(sl); sl.remove()


                self._ghost_trail.append((car.x, car.y))
                if len(self._ghost_trail) > self._MAX_TRAIL:
                    self._ghost_trail.pop(0)

    # ─────────────────────────────────────────────────────────
    #  BLIT: OVERLAYS ON TRACK
    # ─────────────────────────────────────────────────────────
    def _draw_track_overlays(self, best_idx, alive_count):
        ax = self.ax_track


        if self._help_visible:
            self._draw_help_overlay(ax)

        self._draw_notifications(ax)


        _pill(ax, 0.007, 0.018, 0.026, 0.042,
              fc=_alpha(_UI['card'], 0.88), ec=_UI['border_hi'],
              lw=0.8, zorder=19)
        _text(ax, 0.020, 0.040, '?',
              fs=7.5, fw='bold', color=_UI['text2'], ha='center', va='center')

    def _draw_help_overlay(self, ax):
        ow, oh = 0.28, 0.92
        ox, oy = 0.5 - ow/2, 0.5 - oh/2
        _pill(ax, ox, oy, ow, oh,
              fc=_alpha(_UI['card'], 0.97), ec=_UI['border_hi'], lw=1.2, zorder=39)
        _accent_bar(ax, ox, oy, oh, color=_UI['accent'], zorder=40)

        help_lines = [
            ('LEGEND & GLOSSARY',              8, 'bold',   _UI['text1']),
            ('',                               2, 'normal', _UI['text1']),
            ('SIMULATION CARD',                6.5,'bold',  _UI['mid']),
            ('Stagnation   gens without ≥1% improvement',  6, 'normal', _UI['text2']),
            ('              Green<5 | Yellow<15 | Red≥15', 6, 'normal', _UI['text3']),
            ('Avg laps     mean laps last generation',     6, 'normal', _UI['text2']),
            ('Mutation     rate + adaptive mode:',         6, 'normal', _UI['text2']),
            ('  explore   gen<10  high diversity',         6, 'normal', _UI['text3']),
            ('  normal    gen<30  balanced',               6, 'normal', _UI['text3']),
            ('  fine      gen≥30  converging',             6, 'normal', _UI['text3']),
            ('  shake     no gain ×10 gens',               6, 'normal', _UI['accent2']),
            ('  plateau   no gain ×20 gens',               6, 'normal', _UI['danger']),
            ('  CRITICAL  no gain ×40 (max 50% mut)',      6, 'normal', _UI['danger']),
            ('',                               2, 'normal', _UI['text1']),
            ('EVENTS CARD',                    6.5,'bold',  _UI['mid']),
            ('Speed     reward above threshold (20→90%)',  6, 'normal', _UI['text2']),
            ('Lap Done  bonus each completed lap',         6, 'normal', _UI['text2']),
            ('Wall      penalty near wall (post-lap 1)',   6, 'normal', _UI['text2']),
            ('Wrong way penalty going backwards',          6, 'normal', _UI['text2']),
            ('Too slow  threshold escalates each lap',     6, 'normal', _UI['text2']),
            ('',                               2, 'normal', _UI['text1']),
            ('Click ? to close',               6, 'normal', _UI['text3']),
        ]
        cy = oy + oh - 0.022
        for txt, fs, fw, col in help_lines:
            _text(ax, ox + 0.020, cy, txt, fs=fs, fw=fw, color=col,
                  ha='left', va='top', zorder=41)
            cy -= (fs + 1.2) * 0.0055

    def _draw_notifications(self, ax):
        TTL = 3.0; now = time.time()
        self._notifications = [(t, m, p) for t, m, p in self._notifications
                               if now - t < TTL]
        if not self._notifications:
            return

        nt, nm, np_ = self._notifications[-1]
        alpha = max(0.0, 1.0 - (now - nt) / TTL)
        if alpha <= 0:
            return

        nx, ny, nw, nh = 0.862, 0.018, 0.130, 0.058
        _pill(ax, nx, ny, nw, nh,
              fc=_alpha(_UI['card'], alpha * 0.90),
              ec=_alpha(_UI['border_hi'], alpha * 0.8),
              lw=0.7, zorder=19)
        # Red accent
        _rect(ax, (nx, ny), 0.005, nh,
              fc=_UI['danger'], ec='none',
              alpha=alpha * 0.85, zorder=20)
        _text(ax, nx + 0.012, ny + nh - 0.018, nm,
              fs=7.5, fw='bold', color=_alpha(_UI['text1'], alpha),
              ha='left', va='top', zorder=21)
        _text(ax, nx + nw - 0.004, ny + nh - 0.018,
              f'{np_:.0f} pts',
              fs=6.5, color=_alpha(_UI['text2'], alpha),
              ha='right', va='top', zorder=21)

    # ─────────────────────────────────────────────────────────
    #  MAIN UPDATE  (called by timer)
    # ─────────────────────────────────────────────────────────
    def atualizar_visualizacao(self):
        if self.parar_animacao:
            return
        if self._bg_refresh_pending:
            self._bg_refresh_pending = False
            self._ghost_trail.clear()
            self._prepare_background()
            return
        if self.vencedor is not None:
            self._finalizar_treinamento()
            return

        steps = CONFIG['simulation']['steps_per_frame']
        ended = False
        for _ in range(steps):
            ended = self.simular_frame()
            if ended:
                break

        if ended and self.vencedor is None:
            self.evoluir_geracao()
            self.iniciar_tentativa()
            self._bg_refresh_pending = True
            return

        self._frame_count += 1
        self.fig.canvas.restore_region(self._bg)

        alive_mask  = [c.vivo for c in self.carrinhos]
        alive_count = sum(alive_mask)
        alive_idxs  = [i for i, v in enumerate(alive_mask) if v]
        fitness_now = [c.pontos_acumulados for c in self.carrinhos]

        best_idx = (max(alive_idxs, key=lambda i: fitness_now[i])
                    if alive_idxs else None)


        if (self._best_idx_prev is not None
                and not self.carrinhos[self._best_idx_prev].vivo):
            c = self.carrinhos[self._best_idx_prev]
            self._notifications.append(
                (time.time(), c.motivo_morte or 'Eliminated', c.pontos_acumulados))
            self._notifications = self._notifications[-1:]
        self._best_idx_prev = best_idx


        self._draw_cars(best_idx, alive_mask)
        self._draw_track_overlays(best_idx, alive_count)
        self._draw_commands_panel(best_idx)
        self._draw_events_panel(best_idx)
        self._draw_neural_network(best_idx)
        self._draw_lap_panel(best_idx, alive_count)
        self._draw_radar(best_idx)
        self._draw_mini_panel(alive_count)

        self.fig.canvas.blit(self.fig.bbox)

    # ─────────────────────────────────────────────────────────
    #  RESULT SCREEN
    # ─────────────────────────────────────────────────────────
    def _show_result_screen(self, data, _path):
        ax = self.ax_track
        _rect(ax, (0, 0), 1, 1, fc=_alpha(_UI['bg'], 0.78),
              ec='none', transform=ax.transAxes, zorder=29)

        record_str = (f'{data["record_melhor_volta_frames"]} frames'
                      if data['record_melhor_volta_frames'] else '—')
        lines = [
            ('TRAINING COMPLETE',                           22, 'bold',   _UI['accent']),
            ('',                                             5, 'normal', _UI['text1']),
            (f'Generation:   {data["geracao"]}',            13, 'bold',   _UI['text1']),
            (f'Laps:         {data["target_laps"]}',    11, 'normal', _UI['text2']),
            (f'Best lap:     {record_str}',                 11, 'normal', _UI['text2']),
            (f'Winner:       Car #{data["carro_vencedor"]}',11, 'normal', _UI['text2']),
            ('',                                             5, 'normal', _UI['text1']),
            ('Results saved to results.json',               10, 'normal', _UI['accent2']),
            ('',                                             4, 'normal', _UI['text1']),
            ('Close the window to exit.',                    9, 'normal', _UI['text3']),
        ]

   
        cw, ch = 0.35, 0.52
        cx_, cy_ = 0.5 - cw/2, 0.5 - ch/2
        _pill(ax, cx_, cy_, cw, ch,
              fc=_alpha(_UI['card'], 0.96),
              ec=_UI['accent'], lw=1.5, zorder=30,
              transform=ax.transAxes)
        _accent_bar(ax, cx_, cy_, ch, color=_UI['accent'],
                    transform=ax.transAxes, zorder=31)

        y = cy_ + ch - 0.04
        for txt, fs, fw, col in lines:
            ax.text(0.5, y, txt, transform=ax.transAxes,
                    fontsize=fs, ha='center', va='top',
                    fontweight=fw, color=col,
                    fontfamily='monospace', zorder=32)
            y -= 0.046 + (fs - 9) * 0.003
        self.fig.canvas.draw()

    def _finalizar_treinamento(self):
        self.parar_animacao = True
        print('\n' + '═' * 65)
        print(f'  DONE!  Car #{self.vencedor + 1} completed '
              f'{CONFIG["simulation"]["target_laps"]} laps!')
        print(f'  Generation: {self.geracao + 1}')
        if self.record_global_volta:
            print(f'  Best lap:   {self.record_global_volta} frames')
        print('═' * 65)
        path, data = self._salvar_resultados()
        self._show_result_screen(data, path)

    # ─────────────────────────────────────────────────────────
    #  WINDOW EVENTS
    # ─────────────────────────────────────────────────────────
    def _on_key(self, event):
        if event.key == 'q':
            self.parar_animacao = True
            plt.close(self.fig)
        elif event.key == " ":
            vis = CONFIG.setdefault("visualization", {})
            vis["show_sensors"] = not vis.get("show_sensors", True)

    def _on_close(self, _):
        self.parar_animacao = True

    def _on_click(self, event):
        if event.inaxes != self.ax_track:
            return


        disp = self.ax_track.transAxes.transform([[0.007, 0.018], [0.033, 0.060]])
        x0, y0 = disp[0]; x1, y1 = disp[1]
        if x0 <= event.x <= x1 and y0 <= event.y <= y1:
            self._help_visible = not self._help_visible

    # ─────────────────────────────────────────────────────────
    #  START
    # ─────────────────────────────────────────────────────────
    def start_animation(self):
        print('═' * 60)
        print('  NEURAL CIRCUIT — AI Learning to Drive')
        print('═' * 60)
        print(f'\n  GOAL  {CONFIG["simulation"]["target_laps"]} laps without leaving the track')
        print('  GREY  = Asphalt (valid area)')
        print('  GREEN = Off track (grass)')
        print('\n  Q      quit    SPACE  toggle sensors')
        print('═' * 60)

        try:
            mgr = self.fig.canvas.manager

            try:
                mgr.window.state('zoomed')
            except Exception:
                try:
                    mgr.window.showMaximized()
                except Exception:
                    try:
                        mgr.full_screen_toggle()
                    except Exception:
                        pass
        except Exception:
            pass

        self._prepare_background()
        self._bg_refresh_pending = True

        timer = self.fig.canvas.new_timer(interval=0)
        timer.add_callback(self.atualizar_visualizacao)
        timer.start()
        plt.show()