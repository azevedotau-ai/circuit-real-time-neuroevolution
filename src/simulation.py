import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Optional

from .track import (
    CONFIG, _COR, _CENTERLINE,
    AICar, track_tangent,
    nearest_centerline_index,
)
from .neural_network import CarBrain

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  CONSTANTS  (resolved once, not per-frame)
# ─────────────────────────────────────────────────────────────
_CL_TOTAL = len(_CENTERLINE)


# ─────────────────────────────────────────────────────────────
#  MUTATION SCHEDULE
# ─────────────────────────────────────────────────────────────
# Each entry: (stagnation_threshold, mutation_rate, n_random, top_n, mode_label)
_MUT_SCHEDULE = [
    (60, 0.50, 15,  8,  'CRITICAL'),
    (30, 0.38, None, 6, 'plateau'),
    (15, 0.28, None, None, 'shake'),
]


# ─────────────────────────────────────────────────────────────
#  SIMULATOR BASE
# ─────────────────────────────────────────────────────────────
class SimulatorBase:
    """
    Headless neuroevolution training loop.

    Genetic algorithm overview
    ──────────────────────────
    1. Each generation: run cars until all are dead or a winner is found.
    2. Rank by adjusted fitness (score × lap-time bonus).
    3. Elite carry-over + crossover children + random newcomers.
    4. Adaptive mutation rate that increases on stagnation.
    """

    def __init__(self):
        cfg_sim = CONFIG['simulation']

        self.population_size: int           = cfg_sim['population']
        self.cars:  list[AICar]             = []
        self.brains: list[CarBrain]         = []

        self.generation: int                = 0
        self.current_frame: int             = 0
        self.best_fitness_history: list     = []
        self.avg_fitness_history:  list     = []
        self.all_time_best_fitness: float   = 0.0
        self.global_lap_record: Optional[int] = None
        self.global_lap_record_gen: Optional[int] = None
        self.winner: Optional[int]          = None


        self.stagnation_gens: int           = 0
        self._last_best_fitness: float      = 0.0
        self._best_cl_index: int            = 0
        self.current_mut_rate: float        = 0.50
        self.mutation_mode: str             = 'explore'
        self.avg_laps_gen: float            = 0.0

        self.stop: bool                     = False

        self._init_population()

    # ── Legacy attribute aliases ──────────────────────────────
    @property
    def carrinhos(self):         return self.cars
    @property
    def cerebros(self):          return self.brains
    @property
    def geracao(self):           return self.generation
    @property
    def frame_atual(self):       return self.current_frame
    @property
    def historico_melhor(self):  return self.best_fitness_history
    @property
    def historico_media(self):   return self.avg_fitness_history
    @property
    def melhor_fitness_geral(self): return self.all_time_best_fitness
    @property
    def record_global_volta(self):  return self.global_lap_record
    @record_global_volta.setter
    def record_global_volta(self, v): self.global_lap_record = v
    @property
    def record_global_volta_geracao(self): return self.global_lap_record_gen
    @record_global_volta_geracao.setter
    def record_global_volta_geracao(self, v): self.global_lap_record_gen = v
    @property
    def vencedor(self):             return self.winner
    @vencedor.setter
    def vencedor(self, v):          self.winner = v
    @property
    def gens_sem_melhora(self):     return self.stagnation_gens
    @property
    def taxa_mutacao_atual(self):   return self.current_mut_rate
    @property
    def modo_mutacao(self):         return self.mutation_mode
    @property
    def parar_animacao(self):       return self.stop
    @parar_animacao.setter
    def parar_animacao(self, v):    self.stop = v
    @property
    def populacao_size(self):       return self.population_size

    # ─────────────────────────────────────────────────────────
    #  INITIALISATION
    # ─────────────────────────────────────────────────────────
    def _init_population(self, brains: Optional[list] = None):

        n = self.population_size
        self.cars   = [AICar(i, n) for i in range(n)]
        self.brains = brains if brains else [CarBrain() for _ in range(n)]

    def iniciar_tentativa(self):

        for car in self.cars:
            car.reset()
        self.current_frame = 0

    # ─────────────────────────────────────────────────────────
    #  MAIN SIMULATION STEP
    # ─────────────────────────────────────────────────────────
    def simular_frame(self) -> bool:
        """
        Advance all cars by one physics frame.
        Returns True when the attempt is over (all dead or winner found).
        """
        cfg_sim  = CONFIG['simulation']
        rec_cfg  = CONFIG['rewards']
        pen_cfg  = CONFIG['penalties']
        obj_laps = cfg_sim['target_laps']

        alive_indices: list[int]       = []
        alive_sensors: list[np.ndarray] = []


        for idx, car in enumerate(self.cars):
            if not car.alive:
                continue

            # Winner check
            if car.laps >= obj_laps:
                self.winner = idx
                return True

            sensors = car.get_sensors()
            car._cached_sensors = sensors
            car.frame_state     = {}
            car._score_frame_start = car.score

            self._apply_wrong_way(car, pen_cfg)

            if self._apply_stagnation_kill(car, cfg_sim):
                continue

            self._apply_sensor_rewards(car, sensors, rec_cfg, pen_cfg)

            alive_indices.append(idx)
            alive_sensors.append(sensors)


        if alive_indices:
            sensor_matrix = np.stack(alive_sensors)                        
            active_brains = [self.brains[i] for i in alive_indices]
            actions_batch = CarBrain.forward_batch(active_brains, sensor_matrix)  

            for k, idx in enumerate(alive_indices):
                car    = self.cars[idx]
                action = actions_batch[k]

                car.last_action = action
                car.step(action)                          

                self._update_lap_record(car)
                self._apply_progress_reward(car, rec_cfg)
                car.check_collision()


                if car.frame_state.get('wrong_way') and hasattr(car, '_score_frame_start'):
                    car.score = min(car.score, car._score_frame_start)

        self.current_frame += 1
        alive_count = sum(1 for c in self.cars if c.alive)
        return alive_count == 0

    # ─────────────────────────────────────────────────────────
    #  PER-FRAME HELPERS
    # ─────────────────────────────────────────────────────────
    def _apply_wrong_way(self, car: AICar, pen_cfg: dict):
        """
        Detect reverse driving by comparing heading to track tangent.
        Pre-lap: freeze speed.  Post-lap: deduct penalty scaled by alignment.
        """
        rad     = np.radians(car.angle)
        car_dir = np.array([np.cos(rad), np.sin(rad)])
        tangent = track_tangent(car.x, car.y)
        dot     = float(np.dot(car_dir, tangent))

        if dot < pen_cfg['wrong_way_threshold']:
            car.frame_state['wrong_way'] = True
            if car.laps == 0:
                car.speed = 0.0
                _log_event_once(car, 'Freeze – wrong way!', _COR['yellow'])
            else:
                penalty = pen_cfg['wrong_way_penalty'] * max(0.0, -dot)
                car.score -= penalty
                _log_event_once(car, f'−{penalty:.1f} Wrong way!', _COR['red'])

    def _apply_stagnation_kill(self, car: AICar, cfg_sim: dict) -> bool:
        """
        Kill car if centerline progress falls below the minimum threshold
        within the rolling checkpoint window.  Returns True if killed.
        """
        car.checkpoint_frames += 1
        if car.checkpoint_frames < cfg_sim['frames_without_progress']:
            return False

        advance = car.max_cl_index - car.cl_checkpoint
        if advance < cfg_sim['min_progress_per_window']:
            car.alive        = False
            car.death_reason = 'No progress'
            car.events.append(('Stagnant', _COR['red']))
            return True

        car.cl_checkpoint     = car.max_cl_index
        car.checkpoint_frames = 0
        return False

    def _apply_sensor_rewards(self, car: AICar, sensors: np.ndarray,
                              rec_cfg: dict, pen_cfg: dict):
        """
        Reward centerline adherence (minimum sensor × speed).
        Penalise for proximity to walls after the first lap.
        """
        vel_norm   = car.speed / CONFIG['cars']['max_speed']
        min_sensor = float(sensors[:7].min())


        car.score += min_sensor * vel_norm * rec_cfg['center_weight']

        if car.laps > 0:
            frontal_min = float(sensors[2:5].min())
            threshold   = CONFIG['track']['track_width'] / 15.0
            if frontal_min < threshold:
                t       = (threshold - frontal_min) / max(threshold, 1e-9)
                penalty = pen_cfg['near_wall_penalty'] * t
                car.score -= penalty
                car.frame_state['wall'] = True
                _log_event_once(car, f'−{penalty:.2f} Wall!', _COR['red'])

    def _update_lap_record(self, car: AICar):

        if not car.frame_state.get('lap') or car.best_lap_frames is None:
            return
        t = car.best_lap_frames
        if self.global_lap_record is None or t < self.global_lap_record:
            self.global_lap_record     = t
            self.global_lap_record_gen = self.generation + 1
            CONFIG['rewards']['target_lap_frames'] = t
            log.info(f'[gen {self.generation+1}] New lap record: {t} frames')

    def _apply_progress_reward(self, car: AICar, rec_cfg: dict):
        """
        Award points proportional to centerline segments advanced this frame,
        scaled by speed to encourage maintaining pace.
        """
        cl_idx   = nearest_centerline_index(car.x, car.y)
        prev_idx = car.max_cl_index % _CL_TOTAL
        delta    = (cl_idx - prev_idx) % _CL_TOTAL

        if 0 < delta < _CL_TOTAL // 2:
            vel_norm  = car.speed / CONFIG['cars']['max_speed']
            fv        = rec_cfg['speed_progress_factor']
            vel_mult  = 1.0 - fv + fv * vel_norm
            pts       = delta * rec_cfg['circuit_progress_weight'] * vel_mult
            car.score        += pts
            car.max_cl_index += delta

            if delta >= 3:
                car.events.append((f'+{pts:.0f} Progress', _COR['green']))

    # ─────────────────────────────────────────────────────────
    #  GENERATION EVOLUTION
    # ─────────────────────────────────────────────────────────
    def evoluir_geracao(self):

        fitness  = self._adjusted_fitness()
        best     = float(max(fitness))
        avg      = float(np.mean(fitness))

        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)
        if best > self.all_time_best_fitness:
            self.all_time_best_fitness = best

       
        best_cl = max(c.max_cl_index for c in self.cars)
        cl_improved = best_cl > self._best_cl_index
        if cl_improved:
            self._best_cl_index = best_cl

      
        threshold_improve = self._last_best_fitness * 1.005
        if best > threshold_improve or cl_improved:
            self.stagnation_gens     = 0
            self._last_best_fitness  = best
        else:
            self.stagnation_gens += 1

        
        self.avg_laps_gen = float(np.mean([c.laps for c in self.cars]))


        lap_times = [c.best_lap_frames for c in self.cars if c.best_lap_frames is not None]
        if lap_times:
            gen_best = min(lap_times)
            if self.global_lap_record is None or gen_best < self.global_lap_record:
                self.global_lap_record     = gen_best
                self.global_lap_record_gen = self.generation + 1


        rate, n_random, top_n, mode = self._mutation_params()
        self.current_mut_rate = rate
        self.mutation_mode    = mode


        ranked      = np.argsort(fitness)[::-1][:top_n]
        new_brains  = self._build_next_generation(ranked, rate, n_random)
        self._init_population(brains=new_brains)
        self.generation += 1

        log.info(
            f'Gen {self.generation}  best={best:.1f}  avg={avg:.1f}  '
            f'stag={self.stagnation_gens}  mut={rate:.2f} [{mode}]'
        )

    # ─────────────────────────────────────────────────────────
    #  FITNESS  &  SELECTION
    # ─────────────────────────────────────────────────────────
    def _adjusted_fitness(self) -> list[float]:
        """
        Adjust raw score by lap-time quality.
        Cars that completed laps faster than the current record get a boost;
        slower cars are scaled down — incentivises both exploration and speed.
        """
        target = (self.global_lap_record
                  or CONFIG['rewards']['target_lap_frames'])
        fitness = []
        for car in self.cars:
            f = car.score
            if car.best_lap_frames is not None and car.laps >= 1:
                factor = target / max(car.best_lap_frames, 1)
                f *= max(0.4, min(2.5, factor))
            fitness.append(f)
        return fitness

    def _mutation_params(self) -> tuple[float, int, int, str]:
        """
        Adaptive mutation schedule.
        Returns (rate, n_random_newcomers, top_n_survivors, mode_label).
        """
        stag    = self.stagnation_gens
        cfg_sim = CONFIG['simulation']
        top_n   = cfg_sim['top_survivors']
        n_rand  = cfg_sim['new_random_per_generation']

        if stag >= 60: return 0.50, 15,                            8,     'CRITICAL'
        if stag >= 30: return 0.38, min(12, self.population_size // 4), 6, 'plateau'
        if stag >= 15: return 0.28, n_rand,                        top_n, 'shake'
        if self.generation < 10: return 0.40, n_rand,              top_n, 'explore'
        if self.generation < 30: return 0.28, n_rand,              top_n, 'normal'
        return 0.18, n_rand, top_n, 'fine'

    def _build_next_generation(self, elite_indices: np.ndarray,
                               rate: float, n_random: int) -> list:
        """
        Compose next-gen brains:
          1. Elite carry-over (exact copies, no mutation).
          2. Random newcomers (fresh diversity injection).
          3. Crossover children from the elite pool, then mutated.
        """
        new_brains: list[CarBrain] = []
        n_elite = len(elite_indices)

        # 1. Elites
        for idx in elite_indices:
            brain = CarBrain()
            brain.copy_from(self.brains[idx])
            new_brains.append(brain)

        # 2. Random newcomers
        for _ in range(n_random):
            new_brains.append(CarBrain())

        # 3. Children
        while len(new_brains) < self.population_size:
            p1_idx = elite_indices[np.random.randint(0, n_elite)]
            child  = CarBrain()
            child.copy_from(self.brains[p1_idx])


            if n_elite >= 2 and np.random.rand() < 0.5:
                p2_idx = p1_idx
                while p2_idx == p1_idx:
                    p2_idx = elite_indices[np.random.randint(0, n_elite)]
                child.crossover(self.brains[p2_idx])

            child.mutate(rate)
            new_brains.append(child)

        return new_brains[:self.population_size]

    # ─────────────────────────────────────────────────────────
    #  RESULTS PERSISTENCE
    # ─────────────────────────────────────────────────────────
    def _salvar_resultados(self) -> tuple[str, dict]:
        """Save winner stats and network weights to results.json."""
        winner_car   = self.cars[self.winner]
        winner_brain = self.brains[self.winner]

        data = {
            'timestamp':        datetime.now().isoformat(timespec='seconds'),
            'generation':       self.generation + 1,
            'target_laps':      CONFIG['simulation']['target_laps'],
            'best_lap_frames':  self.global_lap_record,
            'winner_car':       self.winner + 1,
            'winner_score':     float(winner_car.score),
            'stagnation_gens':  self.stagnation_gens,
            'neural_network': {
                'W1': winner_brain.W1.tolist(),
                'b1': winner_brain.b1.tolist(),
                'W2': winner_brain.W2.tolist(),
                'b2': winner_brain.b2.tolist(),
            },
            'config_snapshot': {
                'population':   CONFIG['simulation']['population'],
                'top_survivors': CONFIG['simulation']['top_survivors'],
                'mutation_rate': self.current_mut_rate,
                'mutation_mode': self.mutation_mode,
            },
            'fitness_history': {
                'best': self.best_fitness_history,
                'avg':  self.avg_fitness_history,
            },
        }

        path = os.path.join(os.path.dirname(__file__), '..', 'results.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        log.info(f'[results] Saved to {path}')
        return path, data


# ─────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────
def _log_event_once(car: AICar, message: str, color: str):
    if not car.events or car.events[-1][0] != message:
        car.events.append((message, color))



SimuladorBase = SimulatorBase