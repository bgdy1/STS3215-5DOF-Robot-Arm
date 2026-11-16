from contextlib import redirect_stdout
from io import StringIO
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import arm_settings as settings


generator_module = runpy.run_path(
    "scripts/05_waypoint_sequence.py", run_name="action_generator_test"
)


class ActionGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.port = Mock()
        self.packet_handler = object()
        self.read = Mock()
        self.disable = Mock()
        self.overrides = patch.dict(
            generator_module["generate_actions"].__globals__,
            {
                "MOTION_FOLDER": self.path,
                "connect": Mock(return_value=(self.port, self.packet_handler)),
                "disable_all_torque": self.disable,
                "read_all": self.read,
            },
        )
        self.overrides.start()
        self.addCleanup(self.overrides.stop)
        self.first_pose = dict(zip(settings.JOINT_NAMES, (2048, 2100, 2200, 2300, 2400)))
        self.second_pose = dict(zip(settings.JOINT_NAMES, (2000, 2050, 2150, 2250, 2350)))

    def run_generator(self, answers):
        with patch("builtins.input", side_effect=answers):
            return generator_module["generate_actions"]()

    def test_each_q_saves_immediately_to_one_playable_file_one_pose_per_line(self):
        self.read.side_effect = [self.first_pose, self.second_pose]

        def answer(_prompt):
            if answer.index == 0:
                self.assertEqual(list(self.path.iterdir()), [])
                self.disable.assert_not_called()
                reply = "pick_up"
            elif answer.index == 1:
                reply = ""  # Support the arm before torque is released.
            elif answer.index == 2:
                self.assertEqual(list(self.path.iterdir()), [])
                reply = "q"
            else:
                files = list(self.path.glob("*.json"))
                self.assertEqual(len(files), 1)
                saved = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(len(saved["points"]), answer.index - 2)
                reply = "q" if answer.index == 3 else "e"
            answer.index += 1
            return reply

        answer.index = 0
        output = StringIO()
        with redirect_stdout(output):
            path = self.run_generator(answer)
        self.assertEqual(path, self.path / "pick_up.json")
        data, poses = generator_module["load_sequence"](path)
        self.assertEqual(poses, [self.first_pose, self.second_pose])
        self.assertEqual(data["name"], path.stem)
        self.assertEqual(data["input_type"], "angles")
        self.assertEqual(
            data["points"],
            [
                {"input_values": ["0.00", "4.57", "13.36", "22.15", "-30.94"]},
                {"input_values": ["-4.22", "0.18", "8.96", "17.75", "-26.54"]},
            ],
        )
        self.assertIn("angle=", output.getvalue())
        self.assertNotIn("raw=", output.getvalue())
        point_lines = [line for line in path.read_text().splitlines() if '"input_values"' in line]
        self.assertEqual(len(point_lines), 2)
        self.disable.assert_called_once_with(self.packet_handler)
        self.port.closePort.assert_called_once()

    def test_finish_without_capture_creates_no_file(self):
        self.assertIsNone(self.run_generator(["unused", "", "e"]))
        self.assertEqual(list(self.path.iterdir()), [])
        self.read.assert_not_called()
        self.port.closePort.assert_called_once()

    def test_bad_commands_and_failed_reads_do_not_add_points(self):
        self.read.side_effect = [
            RuntimeError("servo timeout"),
            {**self.first_pose, "wrist": -1},
            {**self.first_pose, "base_yaw": 4095},
            self.second_pose,
        ]
        path = self.run_generator(["retry", "", "bad", "", "q", "q", "q", " Q ", "e"])
        data = json.loads(path.read_text())
        self.assertEqual(len(data["points"]), 1)
        self.assertEqual(
            data["points"][0]["input_values"],
            ["-4.22", "0.18", "8.96", "17.75", "-26.54"],
        )

    def test_interrupt_preserves_saved_pose_and_closes_port(self):
        self.read.return_value = self.first_pose
        for stop in (KeyboardInterrupt, EOFError):
            with self.subTest(stop=stop):
                self.port.reset_mock()
                path = self.run_generator([stop.__name__, "", "q", stop()])
                self.assertEqual(len(json.loads(path.read_text())["points"]), 1)
                self.port.closePort.assert_called_once()

    def test_invalid_and_existing_names_are_rejected_before_capture(self):
        self.read.return_value = self.first_pose
        first = self.run_generator(["original", "", "q", "e"])
        original = first.read_bytes()
        second = self.run_generator(
            ["", "../escape", ".json", "x" * 49, "original.json", "new_file.json", "", "q", "e"]
        )
        self.assertEqual(second, self.path / "new_file.json")
        self.assertEqual(json.loads(second.read_text())["name"], "new_file")
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(len(list(self.path.glob("*.json"))), 2)

    def test_failed_save_keeps_previous_json_and_closes_port(self):
        self.read.side_effect = [self.first_pose, self.second_pose]
        generator_globals = generator_module["generate_actions"].__globals__
        real_replace = generator_globals["os"].replace
        calls = 0

        def replace(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk write failed")
            real_replace(source, target)

        with patch.object(generator_globals["os"], "replace", side_effect=replace):
            with self.assertRaisesRegex(OSError, "disk write failed"):
                self.run_generator(["save_error", "", "q", "q"])
        path, = self.path.glob("*.json")
        saved = json.loads(path.read_text())
        self.assertEqual(len(saved["points"]), 1)
        self.port.closePort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
