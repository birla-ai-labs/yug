# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight check: can this machine load and run Yug?

    python -m yug          # or: yug-check

Reports Python, PyTorch, accelerator, memory and disk, then says whether a
forecast will run and how fast to expect it to be. Exits non-zero if something
would actually prevent the model from loading, so it is usable in CI and by
agents as a gate before a long download.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys

# Weights are fp32; ~271.8M parameters is ~1.0 GB on disk and in memory, and
# activations plus the rollout KV cache need headroom on top.
WEIGHTS_GB = 1.15
MIN_RAM_GB = 3.0
RECOMMENDED_VRAM_GB = 4.0
MIN_DISK_GB = 2.0
MIN_PYTHON = (3, 10)


def _gb(n_bytes: float) -> float:
    return n_bytes / (1024**3)


def _host_ram_gb() -> float | None:
    """Total system RAM, or None when it cannot be determined portably."""
    try:
        return _gb(os_sysconf_ram())
    except Exception:
        return None


def os_sysconf_ram() -> float:
    import os

    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yug-check",
        description="Check whether this machine can load and run Yug.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print only the final verdict"
    )
    args = parser.parse_args(argv)

    problems: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    def report(label: str, value: str) -> None:
        lines.append(f"  {label:<22} {value}")

    lines.append("Yug pre-flight check")
    lines.append("=" * 46)

    # --- Python ------------------------------------------------------------
    py = sys.version_info
    report("Python", f"{py.major}.{py.minor}.{py.micro}")
    report("Platform", f"{platform.system()} {platform.machine()}")
    if (py.major, py.minor) < MIN_PYTHON:
        problems.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, found "
            f"{py.major}.{py.minor}"
        )

    # --- PyTorch and accelerator -------------------------------------------
    try:
        import torch
    except ImportError:
        report("PyTorch", "NOT INSTALLED")
        problems.append("PyTorch is not installed: pip install yug")
    else:
        report("PyTorch", torch.__version__)

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            vram = _gb(props.total_memory)
            report("Accelerator", f"CUDA — {props.name}")
            report("VRAM", f"{vram:.1f} GB")
            report("CUDA runtime", torch.version.cuda or "unknown")
            if vram < RECOMMENDED_VRAM_GB:
                warnings.append(
                    f"{vram:.1f} GB of VRAM is below the {RECOMMENDED_VRAM_GB:.0f} GB "
                    f"recommended; reduce num_samples if you hit OOM"
                )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            report("Accelerator", "Apple MPS")
            warnings.append("running on MPS: supported, but noticeably slower than CUDA")
        else:
            report("Accelerator", "none — CPU only")
            warnings.append(
                "no accelerator found; a 512-step forecast takes minutes on CPU "
                "rather than under a second on a GPU. Lower num_samples to compensate."
            )

    # --- Package -----------------------------------------------------------
    try:
        from yug import __version__

        report("yug", __version__)
    except ImportError:
        report("yug", "NOT IMPORTABLE")
        problems.append("yug cannot be imported: pip install yug")

    # --- Memory and disk ---------------------------------------------------
    ram = _host_ram_gb()
    if ram is not None:
        report("System RAM", f"{ram:.1f} GB")
        if ram < MIN_RAM_GB:
            problems.append(
                f"{ram:.1f} GB of RAM is below the {MIN_RAM_GB:.0f} GB needed to "
                f"hold the weights"
            )

    free_disk = _gb(shutil.disk_usage(".").free)
    report("Free disk", f"{free_disk:.1f} GB")
    report("Weights need", f"~{WEIGHTS_GB:.2f} GB")
    if free_disk < MIN_DISK_GB:
        problems.append(
            f"{free_disk:.1f} GB free is not enough to cache the weights "
            f"(need ~{MIN_DISK_GB:.0f} GB with headroom)"
        )

    # --- Verdict -----------------------------------------------------------
    lines.append("")
    for w in warnings:
        lines.append(f"  ! {w}")
    for p in problems:
        lines.append(f"  x {p}")
    if not warnings and not problems:
        lines.append("  All checks passed.")

    lines.append("")
    verdict = (
        "BLOCKED — fix the items marked x before loading the model."
        if problems
        else "READY — this machine can load and run Yug."
    )
    lines.append(verdict)

    if args.quiet:
        print(verdict)
    else:
        print("\n".join(lines))

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
