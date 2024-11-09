import glob

for receptor in ("ER", "PR"):
    best_per_metric: dict[str, tuple[float, str]] = {}
    worst_per_metric: dict[str, tuple[float, str]] = {}
    for path in glob.glob(f"l2_{receptor}*.metrics"):
        with open(path, "r") as f:
            averages = False
            for line in f.read().splitlines():
                if line == "μ":
                    averages = True
                    continue
                if line == "σ":
                    break
                if not averages:
                    continue

                split_line = line.split(": ")
                metric_name = split_line[0]
                metric_val = float(split_line[1])
                if metric_name == "Train time":
                    metric_val *= -1

                if best_per_metric.get(metric_name, (float("-inf"),))[0] < metric_val:
                    best_per_metric[metric_name] = (metric_val, path)
                if worst_per_metric.get(metric_name, (float("+inf"),))[0] > metric_val:
                    worst_per_metric[metric_name] = (metric_val, path)

    print(best_per_metric)
    print(worst_per_metric)
