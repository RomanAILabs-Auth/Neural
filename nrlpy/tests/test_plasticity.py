from nrlpy.plasticity import plasticity_snapshot


def test_sovereign_default() -> None:
    s = plasticity_snapshot(None)
    assert s["writes_enabled"] is False
    assert s["mode"] == "sovereign"


def test_adaptive_stub() -> None:
    s = plasticity_snapshot("adaptive")
    assert s["writes_enabled"] is False
    assert "stub" in s["detail"].lower()
