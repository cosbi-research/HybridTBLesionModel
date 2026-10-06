#!/usr/bin/env python3
"""Run one of the publication model workflows from a single entry point."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "diffusion-profile": ROOT / "src" / "plot_diffusion_profile.py",
    "radial-drug": ROOT / "src" / "simulation_1d.py",
    "circular-abm": ROOT / "src" / "stochastic.py",
    "irregular-drug": ROOT / "src" / "diffusion_manual_domain.py",
}


def project_relative(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative to the release folder")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"{label} must stay inside the release folder")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=SCRIPTS)
    parser.add_argument("--drug", choices=("BDQ", "TBAJ587"), default="TBAJ587")
    parser.add_argument("--output-dir", default="results", help="Release-relative output folder")
    parser.add_argument("--geometry", help="Release-relative boundary CSV for irregular-drug; omit to draw interactively")
    parser.add_argument("--image", help="Release-relative background image for interactive domain drawing")
    parser.add_argument("--simulations", type=int, help="Number of PK variability or ABM replicates")
    parser.add_argument("--diameter", type=float, help="Granuloma diameter in micrometers (radial-drug or circular-abm)")
    parser.add_argument("--donuts", type=int, help="Number of averaging regions (radial-drug or circular-abm)")
    parser.add_argument("--days", type=int, help="Simulation duration in days")
    parser.add_argument("--observations", action="store_true", help="Overlay embedded LCM measurements (radial-drug only)")
    parser.add_argument("--pk-variability", action="store_true", help="Sample PK parameters in drug workflows")
    parser.add_argument("--seed", type=int, help="Random seed for circular ABM")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected script and paths without running it")
    args = parser.parse_args()

    try:
        output_dir = project_relative(args.output_dir, "--output-dir")
        geometry = project_relative(args.geometry, "--geometry") if args.geometry else None
        image = project_relative(args.image, "--image") if args.image else None
    except ValueError as exc:
        parser.error(str(exc))
    if args.simulations is not None and args.simulations < 1:
        parser.error("--simulations must be positive")
    if args.diameter is not None and args.diameter <= 0:
        parser.error("--diameter must be positive")
    if args.donuts is not None and args.donuts not in (4, 5):
        parser.error("--donuts must be 4 or 5")
    if args.days is not None and args.days < 1:
        parser.error("--days must be positive")
    if args.days is not None and args.workflow == "diffusion-profile":
        parser.error("--days does not apply to diffusion-profile")
    if args.days is not None and args.workflow == "circular-abm" and args.days <= 84:
        parser.error("circular-abm needs at least one treated day (more than 84 days)")
    if args.observations and args.workflow != "radial-drug":
        parser.error("--observations applies only to radial-drug")
    if args.pk_variability and args.workflow not in ("radial-drug", "irregular-drug"):
        parser.error("--pk-variability applies only to drug simulations")
    if args.simulations is not None and args.workflow == "diffusion-profile":
        parser.error("--simulations does not apply to diffusion-profile")
    if args.seed is not None and args.workflow != "circular-abm":
        parser.error("--seed applies only to circular-abm")
    if (args.diameter is not None or args.donuts is not None) and args.workflow not in ("radial-drug", "circular-abm"):
        parser.error("--diameter and --donuts apply only to radial-drug or circular-abm")
    if geometry and args.workflow != "irregular-drug":
        parser.error("--geometry applies only to irregular-drug")
    if image and args.workflow != "irregular-drug":
        parser.error("--image applies only to irregular-drug")
    if image and geometry:
        parser.error("--image is used only while drawing a boundary interactively")
    if args.pk_variability and args.workflow == "circular-abm":
        parser.error("--pk-variability applies only to drug workflows")
    if geometry and not (ROOT / geometry).is_file():
        parser.error(f"geometry CSV does not exist: {geometry}")
    if image and not (ROOT / image).is_file():
        parser.error(f"background image does not exist: {image}")

    env = os.environ.copy()
    env.update(TB_DRUG=args.drug, TB_OUTPUT_DIR=str(output_dir))
    if geometry:
        env["TB_GEOMETRY"] = str(geometry)
    if image:
        env["TB_IMAGE"] = str(image)
    if args.simulations is not None:
        env["TB_N_SIM"] = str(args.simulations)
    if args.diameter is not None:
        env["TB_DIAMETER"] = str(args.diameter)
    if args.donuts is not None:
        env["TB_N_DON"] = str(args.donuts)
    if args.days is not None:
        env["TB_T_END"] = str(args.days)
    if args.observations:
        env["TB_PLOT_OBS"] = "1"
    if args.seed is not None:
        env["TB_SEED"] = str(args.seed)
    if args.workflow in ("radial-drug", "irregular-drug"):
        env["TB_PK_VARIABILITY"] = "1" if args.pk_variability else "0"
    if args.workflow != "irregular-drug" or geometry:
        env.setdefault("MPLBACKEND", "Agg")

    script = SCRIPTS[args.workflow]
    print(f"Workflow: {args.workflow}; drug: {args.drug}", flush=True)
    print(f"Script: {script.relative_to(ROOT)}; output: {output_dir}", flush=True)
    if args.dry_run:
        return 0
    (ROOT / output_dir).mkdir(parents=True, exist_ok=True)
    return subprocess.call([sys.executable, str(script)], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
