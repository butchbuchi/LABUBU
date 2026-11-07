"""Importable worker for spawn-based independent runs."""

def run_seed(task):
    run, seed = task
    return run(seed=seed)
