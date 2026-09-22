"""Visual-worker backend: expectimax search with learned frontier values."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter, sleep

import numpy as np
import torch

from .core import slide
from .env import Game2048
from .gpu_search import encode
from .run_io import read_json, write_json
from .value_models import load_value_model
from .value_search import ValueSearch, SearchStopped


@torch.inference_mode()
def score_actions(model, board):
    actions, afterstates = [], []
    for action in range(4):
        after, _, legal = slide(board, action)
        if legal:
            actions.append(action)
            afterstates.append(after)
    scores = [None] * 4
    if not actions:
        return scores, None
    device = next(model.parameters()).device
    inputs = torch.from_numpy(encode(np.stack(afterstates))).to(device)
    predictions = model(inputs).cpu().numpy()
    if not np.isfinite(predictions).all():
        raise ValueError('Model produced nonfinite scores')
    for action, value in zip(actions, predictions):
        scores[action] = float(value)
    best = float(np.max(predictions))
    chosen = next(a for a in actions if scores[a] >= best - 1e-6)
    return scores, chosen


def run(args):
    output = args.output
    started = perf_counter()
    state = dict(status='loading', board=[[0]*4 for _ in range(4)], score=0, step=0,
                 game=1, games=args.games, q=[None]*4, action=None)
    def publish(**fields):
        state.update(fields)
        state['elapsed_seconds'] = perf_counter()-started
        write_json(output / 'status.json', state)
    publish()
    shutil.copyfile(args.checkpoint, output / 'source-model.pt')
    checkpoint = output / 'source-model.pt'
    model = load_value_model(checkpoint, args.device)
    engine = ValueSearch(model, cap=args.search_cap, inference_batch=args.inference_batch,
                         stop=lambda: (output / 'stop.request').exists())
    metadata = dict(checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    model_type=model.model_type,
                    policy='Expectimax; model afterstate frontier; alpha=0; terminal=-10',
                    depth=args.depth, search_cap=args.search_cap, inference_batch=args.inference_batch,
                    tie_atol=1e-6, seed=args.seed, games=args.games, device=args.device)
    write_json(output / 'run.json', metadata)
    publish(model_type=model.model_type, depth=args.depth)
    results, step_token = [], 0
    for game in range(args.games):
        env = Game2048(seed=args.seed+game)
        step, times = 0, []
        game_started = perf_counter()
        publish(status='running', game=game+1, seed=args.seed+game, board=env.board.tolist(),
                score=0, step=0, q=[None]*4, action=None, max_tile=int(env.board.max()))
        with (output / f'game_{game:04d}.jsonl').open('w', encoding='utf-8') as log:
            while not env.terminated:
                if (output / 'stop.request').exists():
                    break
                control = read_json(output / 'control.json', {})
                paused = control.get('paused', args.paused)
                requested = int(control.get('step_token', 0))
                if paused and requested <= step_token:
                    if state['status'] != 'paused':
                        publish(status='paused')
                    sleep(0.03)
                    continue
                step_token = requested
                board = env.board
                before_score = env.score
                tick = perf_counter()
                publish(status='searching')
                try:
                    q, search_stats = engine.search(board[None], depth=args.depth)
                except SearchStopped:
                    break
                values = q[0]
                if not np.isfinite(values).any():
                    raise ValueError('No legal action in a nonterminal game')
                scores = [float(v) if np.isfinite(v) else None for v in values]
                action = next(a for a, v in enumerate(values) if v >= values.max()-1e-6)
                seconds = perf_counter()-tick
                result = env.step(action)
                if not result.changed:
                    raise AssertionError('Model selected an illegal action')
                step += 1
                times.append(seconds)
                record = dict(step=step, board=board.tolist(), next_board=result.board.tolist(),
                              q=scores, action=action, score_before=before_score,
                              depth=args.depth, search_stats=search_stats,
                              score=result.score, reward=result.reward, seconds=seconds,
                              terminated=result.terminated)
                log.write(json.dumps(record)+'\n')
                log.flush()
                publish(status='running', board=result.board.tolist(), evaluated_board=board.tolist(),
                        score=result.score, max_tile=int(result.board.max()), step=step, q=scores,
                        action=action, reward=result.reward, inference_ms=seconds*1000,
                        leaf_evaluations=search_stats['leaf_evaluations'])
                if args.max_steps and step >= args.max_steps:
                    break
                delay = max(0, min(2000, int(control.get('delay_ms', args.delay_ms))))/1000
                until = perf_counter()+delay
                while perf_counter() < until and not (output / 'stop.request').exists():
                    sleep(min(0.03, max(0, until-perf_counter())))
        summary = dict(game=game+1, seed=args.seed+game, steps=step, score=env.score,
                       max_tile=int(env.board.max()), terminated=env.terminated,
                       elapsed_seconds=perf_counter()-game_started,
                       mean_inference_ms=1000*sum(times)/len(times) if times else None)
        results.append(summary)
        write_json(output / 'results.json', results)
        if (output / 'stop.request').exists():
            publish(status='stopped', results=results)
            return
    publish(status='complete', results=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, choices=range(5), default=1,
                        help='Spawn+decision pairs after root action; 0 is direct model scoring')
    parser.add_argument('--search-cap', type=int, default=4096)
    parser.add_argument('--inference-batch', type=int, default=512)
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9200000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--delay-ms', type=int, default=120)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--max-steps', type=int, default=0, help='Bounded test only; 0 plays to terminal')
    args = parser.parse_args(argv)
    if args.games < 1 or args.seed < 0 or args.max_steps < 0 or args.delay_ms < 0:
        parser.error('Invalid game parameters')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output / 'status.json', dict(status='error', message=str(exc)))
        raise


if __name__ == '__main__':
    main()
