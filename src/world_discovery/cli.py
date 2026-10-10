import argparse
import json
from pathlib import Path
import time
from .experiment import collect, train, evaluate, pipeline, manifest, validate_config


def main():
    parser = argparse.ArgumentParser(description='MuJoCo active physical discovery experiments')
    parser.add_argument('command', choices=['run', 'collect', 'train', 'evaluate', 'view'])
    parser.add_argument('--config', default='configs/quick.json')
    parser.add_argument('--out', default='runs/quick-A')
    parser.add_argument('--experiment', choices=['A', 'B'], default='A')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing an existing run directory')
    args = parser.parse_args()
    if args.command == 'view':
        from .viewer import view
        view()
        return
    config = validate_config(json.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.command in ['train', 'evaluate']:
        if args.seeds:
            parser.error('--seeds is supported only for run/collect')
        saved = json.loads((Path(args.out) / 'config.json').read_text())
        protocol = json.loads((Path(args.out) / 'protocol.json').read_text())
        if saved != config or protocol['experiment'] != args.experiment:
            parser.error('Config/experiment differs from collected run; use its config.json and matching experiment')
    seeds = args.seeds if args.seeds is not None else [config['seed']]
    for seed in seeds:
        c = dict(config, seed=seed)
        out = Path(args.out) / f'seed-{seed}' if args.seeds is not None else Path(args.out)
        if args.command in ['run', 'collect'] and out.exists() and any(out.iterdir()) and not args.overwrite:
            parser.error(f'{out} already contains files; choose a new --out or use --overwrite')
        if args.command == 'run':
            pipeline(c, out, args.experiment)
        elif args.command == 'collect':
            collect(c, out, args.experiment)
        elif args.command == 'train':
            train(c, out)
        elif args.command == 'evaluate':
            start = time.perf_counter()
            evaluate(c, out, args.experiment)
            manifest(out, time.perf_counter() - start)


if __name__ == '__main__':
    main()
