"""JSON (de)serialization that safely round-trips numpy arrays and pydantic
models, used by result_store and experiment_registry.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel

_NDARRAY_TAG = "__ndarray__"


class PlatformJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return {_NDARRAY_TAG: True, "dtype": str(o.dtype), "data": o.tolist()}
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, BaseModel):
            return o.model_dump(mode="json")
        return super().default(o)


def _object_hook(d: dict) -> Any:
    if d.get(_NDARRAY_TAG):
        return np.array(d["data"], dtype=d["dtype"])
    return d


def dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(obj, cls=PlatformJSONEncoder, **kwargs)


def loads(s: str, **kwargs: Any) -> Any:
    return json.loads(s, object_hook=_object_hook, **kwargs)
