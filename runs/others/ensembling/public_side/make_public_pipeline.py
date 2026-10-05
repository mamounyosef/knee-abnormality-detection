"""make_public_pipeline.py - the public 0.943 pipeline (yamadan96/rsna-knee-d4-public0946, code by Mattia Angeli, Jiwei
Liu, dreaddevelopment, Pilkwang and others) made PRIZE-SAFE, for the combined notebook.

Changes, each checked to match exactly once:
  1. Stage 3 removed (cells 25-26): RadImageNet ResNet-50 heads (CC-BY-NC-SA, non-commercial) and the 88-feature
     calibrator that needs their ranks. Its datasets are no longer attached.
  2. The final cell no longer demands the calibrator, and writes /kaggle/working/public_side.csv instead of
     submission.csv (our notebook blends it with our ensemble).
  3. TIME_BUDGET reads RSNA_TIME_BUDGET_S (our notebook gives it a cap so our ensemble keeps its share of the 9 h).
  4. find_root() uses RSNA_COMP_ROOT alone when set (the STRESS run points it at a fake 128-study competition root;
     the original also lists the default mount, which would make two roots match).
Everything else is byte-identical, weights stay CC0 (DINOv2-small base is Apache-2.0).

    python runs/others/ensembling/public_side/make_public_pipeline.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "tools_local" / "public_nb" / "yamadan96_rsna-knee-d4-public0946" / "rsna-knee-d4-public0946.ipynb"
OUT = HERE / "public_pipeline.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]
src = ["".join(c["source"]) for c in cells]
assert len(cells) == 31, len(cells)
assert src[25].startswith("## Stage 3") and "_rad_main()" in src[26], "Stage 3 is not cells 25-26"


def patch(i, old, new):
    assert src[i].count(old) == 1, f"cell {i}: {old[:60]!r} found {src[i].count(old)} times"
    src[i] = src[i].replace(old, new)


patch(5, "TIME_BUDGET = 8.0 * 3600", "TIME_BUDGET = float(os.environ.get('RSNA_TIME_BUDGET_S', 8.0 * 3600))  # cap from the combined notebook")
patch(7, "    return _one_direct('competition', [\n"
         "        os.environ.get('RSNA_COMP_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'),\n"
         "        '/kaggle/input/rsna-knee-abnormality-detection',\n"
         "    ], ['test.csv', 'test_series.csv', 'test_series'])",
      "    if os.environ.get('RSNA_COMP_ROOT'):  # combined notebook: an explicit root (the stress run's fake one) wins alone\n"
      "        return _one_direct('competition', [os.environ['RSNA_COMP_ROOT']], ['test.csv', 'test_series.csv', 'test_series'])\n"
      "    return _one_direct('competition', [\n"
      "        os.environ.get('RSNA_COMP_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'),\n"
      "        '/kaggle/input/rsna-knee-abnormality-detection',\n"
      "    ], ['test.csv', 'test_series.csv', 'test_series'])")
patch(30, "if not globals().get('V18_CALIBRATOR_APPLIED'):raise RuntimeError('BTKD calibrator was not applied')\n",
      "# Stage 3 (RadImageNet + its calibrator) removed for prize eligibility: no calibrator check\n")
patch(30, "_final=Path('/kaggle/working/submission.csv')", "_final=Path('/kaggle/working/public_side.csv')")

note = ("# Prize-safe copy for a combined notebook\n\n"
        "Code from `yamadan96/rsna-knee-d4-public0946` (itself credited below), with **Stage 3 removed** "
        "(RadImageNet ResNet-50 heads and their calibrator: RadImageNet is non-commercial). The final cell writes "
        "`public_side.csv`, which the parent notebook rank-blends with its own ensemble. Weights: CC0 datasets; "
        "DINOv2-small (Meta) is Apache-2.0.\n")
keep = [i for i in range(len(cells)) if i not in (25, 26)]
new_cells = [{"cell_type": "markdown", "metadata": {}, "source": note.splitlines(True)}]
for i in keep:
    c = dict(cells[i])
    c["source"] = src[i].splitlines(True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    new_cells.append(c)
nb["cells"] = new_cells
nb["metadata"]["kernelspec"] = {"name": "python3", "language": "python", "display_name": "Python 3"}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
body = "\n".join("".join(c["source"]) for c in new_cells if c["cell_type"] == "code")
for bad in ("_rad_main", "_RAD_CAL_PAYLOAD", "resnet50", "radimagenet-marwan/ResNet50.pt", "V18_CALIBRATOR_APPLIED')"):
    assert bad not in body or bad == "radimagenet-marwan/ResNet50.pt", bad
print(f"wrote {OUT} ({len(new_cells)} cells, {OUT.stat().st_size / 1e3:.0f} kB)")
print("RadImageNet strings left in code (path lists only, never loaded):",
      [s for s in ("radimagenet", "resnet-50-radimagenet") if s in body.lower()])
