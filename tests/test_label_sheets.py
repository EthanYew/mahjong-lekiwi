"""label_sheets.py end to end on a made-up YOLO export: one sheet per name, and a list naming each crop's file."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def make_export(root):
    (root / "test" / "images").mkdir(parents=True)
    (root / "test" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: ['2C', '3C']\n")
    stem = "20261104_101205_capture__001_capture__f3"
    cv2.imwrite(str(root / "test" / "images" / f"{stem}.jpg"), np.full((480, 848, 3), 128, np.uint8))
    (root / "test" / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.03 0.06\n1 0.01 0.01 0.03 0.06\n")


def test_each_name_gets_a_sheet_and_a_corner_box_still_crops(tmp_path, capsys):
    import label_sheets
    folder = tmp_path / "export"
    make_export(folder)
    out = label_sheets.main([str(folder)])
    assert out == tmp_path / "export_sheets"
    sheet = cv2.imread(str(out / "2C.png"))
    width = 2 * label_sheets.HALF_PX * label_sheets.ZOOM * label_sheets.COLUMNS
    assert sheet.shape[1] == width
    assert (out / "3C.png").exists()                     # its box runs off the frame's corner
    listed = (out / "lists.txt").read_text()
    assert "101205 001" in listed and "20261104_101205_capture__001_capture__f3.jpg" in listed
    assert label_sheets.main([str(folder), "--names", "2c,9D", "--out", str(tmp_path / "some")]) == tmp_path / "some"
    assert "9D: no labels" in capsys.readouterr().out
    assert not (tmp_path / "some" / "3C.png").exists()
