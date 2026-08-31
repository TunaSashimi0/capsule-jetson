from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.app.video_worker import CounterSettings, VideoWorker


class VideoWorkerLifecycleTests(unittest.TestCase):
    def test_restart_fails_when_previous_worker_did_not_stop(self) -> None:
        worker = VideoWorker()
        worker._thread = Mock()
        worker._thread.is_alive.return_value = True

        with patch.object(worker, "stop"), patch.object(worker, "start") as start:
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                worker.restart(CounterSettings(model="model.pt"))

        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
