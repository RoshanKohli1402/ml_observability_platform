import pandas as pd
import pytest

from ml_observability import DriftMonitor


class DummyModel:
    def predict_proba(self, X):
        n = len(X)
        # deterministic but simple two-class probabilities
        p1 = pd.Series(range(n), dtype=float) / max(n, 1)
        p0 = 1 - p1
        return pd.concat([p0, p1], axis=1).to_numpy()


def test_import_and_basic_report():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]})
    prod = pd.DataFrame({"a": [2, 3, 4, 5], "b": [2, 2, 3, 3]})

    monitor = DriftMonitor()
    monitor.fit_baseline(train)
    monitor.attach_model(DummyModel())

    report = monitor.monitor(prod)

    assert "global_drift" in report
    assert "confidence_drift" in report
    assert "root_causes" in report
    assert not report["global_drift"].empty


def test_missing_columns_raises():
    train = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    prod = pd.DataFrame({"a": [1, 2, 3]})

    monitor = DriftMonitor()
    monitor.fit_baseline(train)

    with pytest.raises(ValueError, match="missing baseline columns"):
        monitor.monitor(prod)
