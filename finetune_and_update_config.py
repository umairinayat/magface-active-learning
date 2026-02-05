#!/usr/bin/env python
"""
Update config.yaml with a new model checkpoint path.
"""
import argparse
from pathlib import Path
import shutil


def _update_config_weights(config_path: str, new_weights: str, make_backup: bool = True, relative: bool = False):
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", {})
    cfg["model"].setdefault("magface", {})

    if relative:
        new_weights = str(Path(new_weights).resolve().relative_to(cfg_path.parent.resolve()))

    cfg["model"]["magface"]["weights"] = new_weights

    if make_backup:
        backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        shutil.copy2(cfg_path, backup_path)

    tmp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    tmp_path.replace(cfg_path)


def main():
    parser = argparse.ArgumentParser(description="Update config.yaml with a model checkpoint")
    parser.add_argument("--config", default="config.yaml", help="config.yaml to update")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--relative", action="store_true", help="Store weights path relative to config")
    parser.add_argument("--no_backup", action="store_true", help="Do not create config backup")
    args = parser.parse_args()

    _update_config_weights(
        args.config,
        args.weights,
        make_backup=not args.no_backup,
        relative=args.relative,
    )

    print("Updated config weights to:", args.weights)


if __name__ == "__main__":
    main()
