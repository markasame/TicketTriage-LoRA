"""Latency and cost-per-ticket accounting.

Both models in the comparison are 8B-class models served on the same hardware,
so cost per ticket = (mean latency per ticket) x (hardware $/hour). The GPU rate
defaults to RunPod's RTX 4090 community price; override with --gpu-usd-per-hour.
"""

from __future__ import annotations

from statistics import mean, median

DEFAULT_GPU_USD_PER_HOUR = 0.44  # RunPod RTX 4090, community cloud (2026)


def cost_summary(latencies_s: list[float], gpu_usd_per_hour: float = DEFAULT_GPU_USD_PER_HOUR) -> dict:
    if not latencies_s:
        return {"n": 0}
    mean_s = mean(latencies_s)
    return {
        "n": len(latencies_s),
        "mean_latency_s": round(mean_s, 2),
        "median_latency_s": round(median(latencies_s), 2),
        "p95_latency_s": round(sorted(latencies_s)[max(0, int(len(latencies_s) * 0.95) - 1)], 2),
        "gpu_usd_per_hour": gpu_usd_per_hour,
        "usd_per_ticket": round(mean_s / 3600 * gpu_usd_per_hour, 6),
        "usd_per_1k_tickets": round(mean_s / 3600 * gpu_usd_per_hour * 1000, 3),
    }
