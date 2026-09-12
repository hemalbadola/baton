"""The one check that runs today (PRD 6.3, 6.5).

Everything else in tests/unit is skipped until M2. These two assertions are not:
they fail if a constant drifts away from the figure the PRD states, which is the
only way this module can be wrong before it has a body.
"""

from baton.worker.engine import KV_BYTES_PER_LAYER_PER_TOKEN
from baton.worker.probe import OS_RESERVE_BYTES


def test_kv_matches_the_prd_worked_example() -> None:
    """PRD 6.5: 24 layers of 70B at max_len 4096 is 403 MB."""
    total = 24 * 4096 * KV_BYTES_PER_LAYER_PER_TOKEN
    assert round(total / 1e6) == 403


def test_os_reserve_matches_the_prd_table() -> None:
    """PRD 6.3: 1 GB on cuda, 2 GB on mps, 1.5 GB on cpu."""
    gb = 1024**3
    assert OS_RESERVE_BYTES == {"cuda": gb, "mps": 2 * gb, "cpu": 3 * gb // 2}
