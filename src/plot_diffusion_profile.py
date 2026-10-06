"""Plot the fitted spatial diffusion function used in the paper's Figure 3a-b."""

import matplotlib.pyplot as plt
import numpy as np

from release_config import OUTPUT_DIR, selected_drug


# Table 2 of the accompanying paper: D in um^2/day, b in um.
PARAMETERS = {
    "BDQ": (13330, 5538, 37, 1088, 3400),
    "TBAJ587": (9170, 5870, 83, 1400, 4000),
}


def main() -> None:
    drug = selected_drug()
    d_max, d_min, shape, b, fitted_diameter = PARAMETERS[drug]
    distance = np.linspace(0, fitted_diameter / 2, 1001)
    # Spatially varying Hill/Emax coefficient from Methods, Eq. (4).
    diffusion = (d_max - d_min) * (1 - distance**shape / (b**shape + distance**shape)) + d_min
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distance, diffusion, color="tab:blue", linewidth=2)
    ax.set(xlabel="Distance from boundary (µm)", ylabel="Diffusion coefficient (µm²/day)", title=drug)
    ax.grid(alpha=0.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"diffusion_profile_{drug}.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
