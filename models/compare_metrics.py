import glob

import click
from numpy import format_float_positional


@click.command()
@click.argument("pattern")
def main(pattern: str):
    paths = glob.glob(pattern)
    for receptor_paths in (
        (path for path in paths if "ER_" in path),
        (path for path in paths if "PR_" in path),
    ):
        best_per_metric: dict[str, tuple[float, str]] = {}
        worst_per_metric: dict[str, tuple[float, str]] = {}
        for path in receptor_paths:
            with open(path, "r") as f:
                averages = False
                for line in f.read().splitlines():
                    if line == "MEAN":
                        averages = True
                        continue
                    if line == "STD_DEV":
                        break
                    if not averages:
                        continue

                    split_line = line.split(": ")
                    metric_name = split_line[0]
                    metric_val = float(split_line[1])
                    if metric_name == "Train time":
                        metric_val *= -1

                    if (
                        best_per_metric.get(metric_name, (float("-inf"),))[0]
                        < metric_val
                    ):
                        best_per_metric[metric_name] = (metric_val, path)
                    if (
                        worst_per_metric.get(metric_name, (float("+inf"),))[0]
                        > metric_val
                    ):
                        worst_per_metric[metric_name] = (metric_val, path)

        print(
            {
                k: (
                    format_float_positional(
                        v1, precision=3, unique=False, fractional=False
                    ),
                    v2,
                )
                for k, (v1, v2) in best_per_metric.items()
            }
        )
        print(
            {
                k: (
                    format_float_positional(
                        v1, precision=3, unique=False, fractional=False
                    ),
                    v2,
                )
                for k, (v1, v2) in worst_per_metric.items()
            }
        )


if __name__ == "__main__":
    main()
