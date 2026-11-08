from swarm_agent import backends


def test_hardware_report_shape():
    report = backends.hardware_report()
    assert isinstance(report, dict)
    for key in ("system", "machine", "python", "cpu_count",
                "memory_gb", "apple_silicon", "chip", "unified_memory_gb"):
        assert key in report
    assert report["cpu_count"] >= 1


def test_not_apple_silicon_on_linux():
    report = backends.hardware_report()
    if report["system"] == "Linux":
        assert report["apple_silicon"] is False
        assert report["unified_memory_gb"] is None
        assert report["memory_gb"] is not None


def test_recommendations_are_text():
    tips = backends.recommendations()
    assert isinstance(tips, list)
    assert len(tips) >= 1
    assert all(isinstance(t, str) and t for t in tips)
    assert any("LAN" in t for t in tips)
