from __future__ import annotations

import json

import numpy as np
import pytest

from tools.eval.paired_aux_block_bootstrap import json_default


def test_json_default_serializes_numpy_results() -> None:
    payload = {
        "count": np.int64(7),
        "score": np.float32(0.25),
        "taus": np.asarray([1, 2, 3], dtype=np.int8),
    }

    encoded = json.dumps(payload, default=json_default)

    assert json.loads(encoded) == {"count": 7, "score": 0.25, "taus": [1, 2, 3]}


def test_json_default_rejects_unknown_types() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_default(object())
