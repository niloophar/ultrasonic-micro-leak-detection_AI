#!/usr/bin/env python
"""
Standalone Excel export for the "does temperature explain PC1 drift?" check
(scripts/analyze_center_check_pc1_temperature.py) — same content as the
"PC1 vs Temperature" sheet in haus_b_adaptation_comparison.xlsx, split out
into its own file.

Reads:
  data/results/haus_b_adaptation/center_check_pc1_temperature.json

Writes:
  data/results/haus_b_adaptation/pc1_vs_temperature.xlsx

Run: python scripts/export_pc1_temperature_excel.py
"""
from pathlib import Path
import sys

from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_haus_b_adaptation_excel import load, write_pc1_temperature_sheet, OUT_DIR


def main():
    pt_path = OUT_DIR / 'center_check_pc1_temperature.json'
    if not pt_path.exists():
        raise SystemExit(f"{pt_path} not found — run analyze_center_check_pc1_temperature.py first.")

    wb = Workbook()
    wb.remove(wb.active)
    write_pc1_temperature_sheet(wb, load(pt_path))

    out_path = OUT_DIR / 'pc1_vs_temperature.xlsx'
    wb.save(out_path)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
