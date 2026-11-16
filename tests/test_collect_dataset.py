import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from collect_dataset import CaptureUI


class AnnotationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        self.ui = CaptureUI(self.root)
        self.ui.live = np.zeros((720, 1280, 3), np.uint8)
        self.ui.action("freeze")
        self.ui.render()

    def click(self, action):
        self.ui.render()
        rect = next(rect for rect, name in self.ui.buttons if name == action)
        self.ui.mouse(cv2.EVENT_LBUTTONDOWN, rect[0] + 10, rect[1] + 10, 0, None)
        self.ui.mouse(cv2.EVENT_LBUTTONUP, rect[0] + 10, rect[1] + 10, 0, None)

    def drag(self, start=(100, 100), end=(200, 200)):
        self.ui.mouse(cv2.EVENT_LBUTTONDOWN, *start, 0, None)
        self.ui.mouse(cv2.EVENT_MOUSEMOVE, *end, 0, None)
        self.ui.mouse(cv2.EVENT_LBUTTONUP, *end, 0, None)

    def test_two_bricks_are_classified_after_drawing(self):
        e = self.ui.editor
        self.drag()
        self.assertEqual(e.annotations, [])
        self.assertIsNone(e.color)
        self.assertFalse(e.confirm())
        self.click("color:0")
        self.assertFalse(e.confirm())
        self.click("size:0")
        self.click("confirm")
        self.assertEqual(e.annotations[0][0], 0)
        self.drag((300, 300), (400, 400))
        self.assertIsNone(e.color)
        self.assertIsNone(e.size)
        self.click("color:5")
        self.click("size:1")
        self.click("confirm")
        self.assertEqual([c for c, _ in e.annotations], [0, 11])
        self.click("save")
        labels = next((self.root / "labels").glob("*.txt")).read_text().splitlines()
        self.assertEqual([line.split()[0] for line in labels], ["0", "11"])
        image = cv2.imread(str(next((self.root / "images").glob("*.jpg"))))
        self.assertEqual(image.shape, (720, 1280, 3))
        self.assertEqual(int(image.max()), 0)  # Keine UI-Elemente im Trainingsbild.

    def test_pending_box_blocks_save_and_overwriting(self):
        self.drag()
        box = self.ui.editor.pending
        self.drag((300, 300), (400, 400))
        self.assertEqual(self.ui.editor.pending, box)
        self.ui.action("save")
        self.assertEqual(list((self.root / "images").iterdir()), [])
        self.ui.action("cancel")
        self.assertIsNone(self.ui.editor.pending)

    def test_reverse_drag_maps_back_to_original_image(self):
        self.drag((430, 242), (215, 121))
        self.assertEqual(self.ui.editor.pending, (320, 180, 320, 180))

    def test_photo_queue_is_separate_from_labeled_training_images(self):
        self.ui.action("photo")
        self.assertEqual(len(list((self.root / "unlabeled").glob("*.jpg"))), 1)
        self.assertEqual(list((self.root / "images").iterdir()), [])
        self.assertEqual(list((self.root / "labels").iterdir()), [])

    def test_photo_does_not_discard_pending_annotation(self):
        self.drag()
        self.ui.action("photo")
        self.assertIsNotNone(self.ui.editor.pending)
        self.assertFalse((self.root / "unlabeled").exists())



if __name__ == "__main__":
    unittest.main()
