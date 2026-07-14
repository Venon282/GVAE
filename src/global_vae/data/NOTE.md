# Deferred

`datamodule.py` and `transforms/` are deferred pending the open
question in spec §11 ("First concrete joint signal+image
dataset/task"). SAXS-specific preprocessing (e.g. log-scale intensity)
belongs in `transforms/`, not in the encoder (spec §6).
