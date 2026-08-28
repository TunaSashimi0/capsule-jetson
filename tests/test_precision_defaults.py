import unittest

from src.app.video_worker import CounterSettings
from src.capsule_yolo.export_model import build_parser as export_parser
from src.capsule_yolo.infer_video import build_parser as video_parser
from src.capsule_yolo.predict_image import build_parser as predict_parser
from src.capsule_yolo.train import build_parser as train_parser
from src.capsule_yolo.validate import build_parser as validate_parser


class PrecisionDefaultTests(unittest.TestCase):
    def test_cuda_workflows_default_to_mixed_or_half_precision(self) -> None:
        self.assertTrue(train_parser().get_default("amp"))
        self.assertTrue(export_parser().get_default("half"))
        self.assertTrue(video_parser().get_default("half"))
        self.assertTrue(predict_parser().get_default("half"))
        self.assertTrue(validate_parser().get_default("half"))
        self.assertTrue(CounterSettings(model="model.pt").half)


if __name__ == "__main__":
    unittest.main()
