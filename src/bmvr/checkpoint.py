"""Portable array checkpoints with explicit time and model metadata."""

import json
from pathlib import Path
import numpy as np
from .grid import Grid


def save(path, grid, fields, time, dt, metadata=None):
    payload = {"time": float(time), "dt": float(dt), "periodic": grid.periodic,
               "field_names": list(fields), "metadata": metadata or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xe=grid.xe, ye=grid.ye, description=np.array(json.dumps(payload)),
                        **{"field_" + k: v for k, v in fields.items()})


def load(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["description"]))
        grid = Grid(data["xe"].copy(), data["ye"].copy(), meta["periodic"])
        fields = {name: data["field_" + name].copy() for name in meta["field_names"]}
    return grid, fields, meta
