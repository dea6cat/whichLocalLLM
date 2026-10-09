"""Library entry point for programs that already know their machine (ClydeCLI passes its own).

The CLI detects hardware itself; here the caller hands in a `HardwareInfo` and gets the ranked
models back, with the same fetching, caching and scoring as the command line.
"""

from __future__ import annotations

import asyncio

from whichllm.engine.ranker import rank_models
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.artifacts import attach_resolved_artifacts
from whichllm.models.benchmark import (
    fetch_benchmark_scores,
    load_benchmark_cache,
    save_benchmark_cache,
)
from whichllm.models.cache import load_cache, save_cache
from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
from whichllm.models.grouper import group_models


def recommend(
    hardware: HardwareInfo,
    *,
    top: int = 5,
    context_length: int = 4096,
    profile: str = "general",
    quant: str | None = None,
    refresh: bool = False,
) -> list[CompatibilityResult]:
    """The best models for `hardware`, best first. Raises on a failed model fetch; benchmark data is optional."""
    cached = None if refresh else load_cache()
    if cached is not None:
        models = dicts_to_models(cached)
    else:
        models = asyncio.run(fetch_models(include_vision=profile in ("vision", "any")))
        save_cache(models_to_dicts(models))

    scores = None if refresh else load_benchmark_cache()
    if scores is None:
        try:
            scores = asyncio.run(fetch_benchmark_scores())
            save_benchmark_cache(scores)
        except Exception:   # ranking still works without benchmarks, with lower confidence
            scores = {}

    candidates = [m for family in group_models(models) for m in (family.base_model, *family.variants)]
    results = rank_models(
        candidates, hardware, context_length=context_length, top_n=top, quant_filter=quant,
        benchmark_scores=scores, task_profile=profile, require_direct_top=True,
    )
    attach_resolved_artifacts(results, candidates, quant_filter=quant)
    return results
