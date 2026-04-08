import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from results_figures import common


OUTPUT_DIR = common.PROJECT_ROOT / "outputs" / "results_figures_word_png"
EXPORT_DPI = 600


def save_figure_word_png(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=EXPORT_DPI)
    plt.close(fig)
    return path


if __name__ == "__main__":
    common.OUTPUT_DIR = OUTPUT_DIR
    common.save_figure = save_figure_word_png
    output_path = common.create_fig07_optimization_landscape()
    print(f"已导出 Word 版高分辨率 PNG: {output_path}")
