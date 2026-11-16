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
        self.assertEqual(len(summary["unlearned_classes"]), 12)
        self.assertEqual(len(json.loads(config.read_text())["names"]), 13)
        with self.assertRaises(FileExistsError):
            prepare(self.raw, out)

    def test_identical_pixels_across_sessions_are_rejected(self):
        self.session("01", 10)
        self.session("02", 10)
        out = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "Identische Bilder"):
            prepare(self.raw, out)
        self.assertFalse(out.exists())

    def test_legacy_sessions_keep_ids_when_figure_class_is_appended(self):
        old = self.session("01", 10)
        (old / "classes.json").write_text(json.dumps(CLASSES[:12]))
        new = self.session("02", 20)
        (new / "labels/brick.txt").write_text("12 0.5 0.5 0.5 0.5\n")
        _, summary = prepare(self.raw, self.root / "prepared")
        self.assertEqual(summary["class_counts"]["train"], {"maennchen": 1})
        self.assertEqual(summary["class_counts"]["val"], {"gelb_2x2": 1})

    def test_new_session_adds_to_training_without_dropping_older_classes(self):
        self.session("01", 10)
        self.session("02", 20)
        latest = self.session("03", 30)
        (latest / "labels/brick.txt").write_text("12 0.5 0.5 0.5 0.5\n")
        _, summary = prepare(self.raw, self.root / "prepared")
        self.assertEqual(summary["sessions"], {"train": ["02", "03"], "val": ["01"]})
        self.assertEqual(summary["class_counts"]["train"], {"gelb_2x2": 1, "maennchen": 1})
        self.assertEqual(summary["unvalidated_classes"], ["maennchen"])

    def test_explicit_holdout_group_is_not_copied_to_training(self):
        self.session("01", 10)
        latest = self.session("02", 20)
        cv2.imwrite(str(latest / "images/left.png"), np.full((20, 20, 3), 30, np.uint8))
        (latest / "labels/left.txt").write_text("12 0.5 0.5 0.5 0.5\n")
        (latest / "review").mkdir()
        (latest / "review/labeling_report.json").write_text(json.dumps({
            "validation_images": ["left.png"], "split_policy": "Left position group"}))
        out = self.root / "prepared"
        _, summary = prepare(self.raw, out)
        self.assertEqual(summary["image_counts"], {"train": 1, "val": 2})
        self.assertFalse((out / "images/train/02_left.png").exists())
        self.assertTrue((out / "images/val/02_left.png").exists())
        self.assertEqual(summary["heldout_groups"], {"02": "Left position group"})


if __name__ == "__main__":
    unittest.main()
