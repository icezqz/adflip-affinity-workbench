import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
OUT_DIR = ROOT / "outputs" / "figures"


ZERO_SHOT = {
    "Shanehsazzadeh2023_trastuzumab_zero_kd": -0.11945,
    "Warszawski2019_d44_Kd": -0.178839,
    "Koenig2017_g6_Kd": -0.183597,
}

MODEL_FILES = {
    "our_S": CKPT_DIR / "Shanehsazzadeh2023_trastuzumab_zero_kd_20260317_112657-epoch=11-val_loss=0.1824.ckpt",
    "our_W": CKPT_DIR / "Warszawski2019_d44_Kd_20260320_134257-epoch=14-val_loss=0.5960.ckpt",
    "our_K": CKPT_DIR / "Koenig2017_g6_Kd_20260320_151647-epoch=09-val_loss=0.2272.ckpt",
}

DATASETS = [
    "Shanehsazzadeh2023_trastuzumab_zero_kd",
    "Warszawski2019_d44_Kd",
    "Koenig2017_g6_Kd",
]


def eval_json_for(ckpt_path: Path, dataset_name: str) -> Path:
    stem = ckpt_path.name[:-5]
    return ckpt_path.with_name(f"{stem}.eval_{dataset_name}.json")


def load_table() -> List[Dict[str, float]]:
    rows = []
    for dataset_name in DATASETS:
        row = {"dataset": dataset_name, "adflip": ZERO_SHOT[dataset_name]}
        for model_name, ckpt_path in MODEL_FILES.items():
            metrics_path = eval_json_for(ckpt_path, dataset_name)
            with open(metrics_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            row[model_name] = payload["metrics"]["test_spearman"]
        rows.append(row)
    return rows


def save_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    fieldnames = ["dataset", "adflip", "our_S", "our_W", "our_K"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bar_svg(rows: List[Dict[str, float]], out_svg: Path) -> None:
    series = ["adflip", "our_S", "our_W", "our_K"]
    colors = ["#9AA5B1", "#4C78A8", "#F58518", "#54A24B"]
    width = 980
    height = 460
    left = 90
    right = 30
    top = 50
    bottom = 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    ymin = -0.25
    ymax = 0.30

    def y_to_px(val: float) -> float:
        return top + (ymax - val) / (ymax - ymin) * plot_h

    def esc(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    n_group = len(rows)
    group_w = plot_w / n_group
    bar_w = group_w * 0.16

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">Cross-dataset Spearman comparison</text>',
    ]

    for tick in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
        y = y_to_px(tick)
        stroke = "#222222" if abs(tick) < 1e-12 else "#D9D9D9"
        stroke_w = 1.2 if abs(tick) < 1e-12 else 1.0
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{tick:.1f}</text>'
        )

    for gi, row in enumerate(rows):
        group_x = left + gi * group_w
        center_x = group_x + group_w / 2
        for si, (name, color) in enumerate(zip(series, colors)):
            value = float(row[name])
            x = group_x + group_w * 0.12 + si * bar_w * 1.25
            y0 = y_to_px(0.0)
            yv = y_to_px(value)
            rect_y = min(y0, yv)
            rect_h = abs(yv - y0)
            parts.append(
                f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" fill="{color}"/>'
            )
            text_y = rect_y - 6 if value >= 0 else rect_y + rect_h + 14
            parts.append(
                f'<text x="{x + bar_w/2:.1f}" y="{text_y:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{value:.2f}</text>'
            )

        parts.append(
            f'<text x="{center_x:.1f}" y="{height-58}" text-anchor="middle" font-size="11" font-family="Arial">{esc(row["dataset"])}</text>'
        )

    legend_x = left
    legend_y = height - 24
    for idx, (name, color) in enumerate(zip(series, colors)):
        x = legend_x + idx * 120
        parts.append(f'<rect x="{x}" y="{legend_y-10}" width="14" height="14" fill="{color}"/>')
        parts.append(
            f'<text x="{x+20}" y="{legend_y+2}" font-size="12" font-family="Arial">{esc(name)}</text>'
        )

    parts.append(
        f'<text transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" font-family="Arial">Spearman</text>'
    )
    parts.append("</svg>")

    out_svg.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_table()
    save_csv(rows, OUT_DIR / "three_dataset_spearman.csv")
    plot_grouped_bar_svg(rows, OUT_DIR / "three_dataset_spearman.svg")
    for row in rows:
        print(
            row["dataset"],
            f"adflip={row['adflip']:.4f}",
            f"our_S={row['our_S']:.4f}",
            f"our_W={row['our_W']:.4f}",
            f"our_K={row['our_K']:.4f}",
        )
    print(f"Saved CSV: {OUT_DIR / 'three_dataset_spearman.csv'}")
    print(f"Saved SVG: {OUT_DIR / 'three_dataset_spearman.svg'}")


if __name__ == "__main__":
    main()
