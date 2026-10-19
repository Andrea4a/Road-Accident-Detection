from __future__ import annotations

import argparse
import json

from demo_backend import DemoConfig, DemoSystem, load_config


def main():
    parser = argparse.ArgumentParser(description="Inferenza CLI per il dimostratore della tesi")
    parser.add_argument("--video", required=True, help="Percorso del video da analizzare")
    parser.add_argument("--config", help="Percorso del file config.json")
    parser.add_argument("--src-root", help="Cartella src del progetto")
    parser.add_argument("--weights-path", help="Checkpoint .pth")
    parser.add_argument("--val-list", default=None, help="CSV validation opzionale")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Threshold fisso opzionale")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device da usare")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        if not args.src_root or not args.weights_path:
            parser.error("Senza --config devi passare almeno --src-root e --weights-path")
        cfg = DemoConfig(
            src_root=args.src_root,
            weights_path=args.weights_path,
            val_list=args.val_list,
            fixed_threshold=args.fixed_threshold,
            device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        )

    system = DemoSystem(cfg)
    result = system.predict_video(args.video)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
