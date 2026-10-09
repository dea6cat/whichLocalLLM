from __future__ import annotations

from whichllm import api
from whichllm.hardware.types import GPUInfo, HardwareInfo

_GiB = 1024 ** 3


def _mac(vram_gib: int) -> HardwareInfo:
    gpu = GPUInfo("Apple M3 Pro", "apple", vram_gib * _GiB, usable_vram_bytes=vram_gib * _GiB, memory_bandwidth_gbps=150.0, shared_memory=True)
    return HardwareInfo(gpus=[gpu], cpu_name="Apple M3 Pro", cpu_cores=11, ram_bytes=18 * _GiB, os="darwin")


def test_recommend_ranks_for_the_hardware_it_is_given(monkeypatch):
    seen = {}

    def fake_rank(candidates, hardware, **kwargs):
        seen["hardware"], seen["top"] = hardware, kwargs["top_n"]
        return []

    monkeypatch.setattr(api, "load_cache", lambda: [])
    monkeypatch.setattr(api, "dicts_to_models", lambda data: [])
    monkeypatch.setattr(api, "load_benchmark_cache", lambda: {})
    monkeypatch.setattr(api, "rank_models", fake_rank)
    monkeypatch.setattr(api, "attach_resolved_artifacts", lambda *a, **k: None)
    machine = _mac(12)

    assert api.recommend(machine, top=3) == []
    assert seen["hardware"] is machine and seen["top"] == 3
