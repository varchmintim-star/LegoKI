import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from collect_dataset import CLASSES
from prepare_dataset import prepare


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.raw = self.root / "raw"

    def session(self, name, pixel):
        root = self.raw / name
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.json").write_text(json.dumps(CLASSES))
        cv2.imwrite(str(root / "images/brick.png"), np.full((20, 20, 3), pixel, np.uint8))
        (root / "labels/brick.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        return root

    def test_sessions_are_separated_and_flagged_images_excluded(self):
        old = self.session("01", 10)
        new = self.session("02", 20)
        cv2.imwrite(str(old / "images/uncertain.png"), np.zeros((20, 20, 3), np.uint8))
        (old / "review").mkdir()
        (old / "review/labeling_report.json").write_text(json.dumps({
            "exclude_from_training_until_reviewed": {"uncertain.png": "unclear"}}))
        out = self.root / "prepared"
        config, summary = prepare(self.raw, out)
        self.assertEqual(summary["sessions"], {"train": ["02"], "val": ["01"]})
        self.assertEqual(summary["image_counts"], {"train": 1, "val": 1})
        self.assertTrue((out / "images/train/02_brick.png").exists())
        self.assertTrue((out / "labels/val/01_brick.txt").exists())
        self.assertEqual(len(summary["unlearned_classes"]), 11)
        self.assertEqual(len(json.loads(config.read_text())["names"]), 12)
        with self.assertRaises(FileExistsError):
            prepare(self.raw, out)

    def test_identical_pixels_across_sessions_are_rejected(self):
        self.session("01", 10)
        self.session("02", 10)
        out = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "Identische Bilder"):
            prepare(self.raw, out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
