# Contributing

Install `python -m pip install -e '.[baselines,figures]'` and run
`python -m unittest discover -s tests -v` before proposing a change.
For optimizer changes, compare fixed-seed mappings, cost components and checkpoints;
include the input hash and all parameters. Keep raw results separate from new output.
Do not silently change paper parameters, candidate multiplicities or the update schedule.
Report reproducible bugs with a small QASM example and the command used.
