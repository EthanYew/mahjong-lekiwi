import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import detector_eval  # noqa: E402


def test_the_wrong_name_bound_is_the_rule_of_three_at_zero():
    # v6x3's exam (2026-10-28): 99 named, none wrong, still allows ~3%; ~1000 reads are needed for 0.3%
    assert detector_eval.wrong_rate_bound(0, 99) == pytest.approx(0.0298, abs=1e-4)
    assert detector_eval.wrong_rate_bound(0, 1000) == pytest.approx(0.0030, abs=1e-4)
    assert detector_eval.wrong_rate_bound(1, 99) == pytest.approx(0.0470, abs=1e-4)
    assert detector_eval.wrong_rate_bound(5, 5) == 1.0
