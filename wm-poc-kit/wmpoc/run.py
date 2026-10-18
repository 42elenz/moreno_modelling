"""Entry point: python -m wmpoc.run --config <yaml> [dot.overrides ...]"""
import argparse

from wmpoc.config import load_config, to_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    print(to_yaml(load_config(a.config, a.overrides)))


if __name__ == "__main__":
    main()
