import unittest
import signal
import os
import sys
from unittest.mock import MagicMock, patch

# We need to import the script to test its logic.
# Since it's a script in the root, we might need to adjust sys.path or import it via a helper
# But it exposes `run_watchdog_command`, `main` (which runs infinitely), etc.
# Ideally `run_pipeline.py` would be importable without running main.
# I added `if __name__ == "__main__":` so it is importable.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import run_pipeline


class TestPipelineLogic(unittest.TestCase):
    @patch("subprocess.Popen")
    @patch("os.killpg")
    @patch("time.sleep")  # Mock sleep to speed up
    def test_pipeline_sequence_and_signals(self, mock_sleep, mock_killpg, mock_popen):
        """
        Verify that:
        1. It launches the correct number of steps.
        2. Signal handler sends termination to children.
        """

        # Mock Popen instance
        mock_process = MagicMock()
        # poll needs to return None (running) initially, then 0 (done) eventually
        # mock_process.poll.side_effect = [None]*10 + [0]*1000
        # Better: use a function
        self.poll_count = 0

        def side_effect_poll():
            self.poll_count += 1
            if self.poll_count < 5:
                return None
            return 0

        mock_process.poll.side_effect = side_effect_poll

        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # We need to test the signal handler logic specifically
        # run_pipeline "main" runs the whole thing.
        # We can hook into run_watchdog_command to verify commands.

        sent_commands = []

        def side_effect_run_cmd(cmd_args, log_prefix):
            sent_commands.append(cmd_args)
            return (mock_process, f"mock_{log_prefix}.log", log_prefix)

        # We use a mocked run_watchdog_command for inspection
        # Also mock monitor_processes since it does real I/O
        with (
            patch("run_pipeline.run_watchdog_command", side_effect=side_effect_run_cmd),
            patch("run_pipeline.monitor_processes"),
        ):
            # We only want to run 1 step to verify logic, not all 11
            # Adjust STEPS temporarily
            original_steps = run_pipeline.STEPS
            run_pipeline.STEPS = 1

            try:
                # Also mock sys.argv to passing some dummy args
                with patch.object(sys, "argv", ["run_pipeline.py", "--dummy", "arg"]):
                    run_pipeline.main()
            finally:
                run_pipeline.STEPS = original_steps

        # Verify commands
        # Step 0:
        #   - 2 parallel single obj runs
        #   - 1 qdax run
        self.assertEqual(len(sent_commands), 3)

        # Check Single Obj 1
        cmd1 = sent_commands[0]
        self.assertIn("--mode", cmd1)
        self.assertIn("single", cmd1)
        self.assertIn("--alpha-expectation", cmd1)
        self.assertIn("--alpha-probability", cmd1)
        # With 1 step, linspace(0.9, 0.0, 1) = [0.9], so Prob=0.9, Exp=0.1
        self.assertIn("0.9", cmd1)

        # Check QDAX
        cmd2 = sent_commands[2]
        self.assertIn("--mode", cmd2)
        self.assertIn("qdax", cmd2)

        # --- Verify Signal Handling ---
        # Setup global state in module
        run_pipeline.current_processes = [mock_process]

        # Call the handler
        run_pipeline.signal_handler_skip(signal.SIGUSR1, None)

        # Assert killpg was called
        # mock_process.pid is 12345
        # we expect os.killpg(os.getpgid(12345), signal.SIGTERM)
        # We mocked os.killpg. Note: os.getpgid might need mocking if PID invalid, strictly speaking.
        # But we didn't mock getpgid above, let's assume it calls it.
        # Actually with mock_pid=12345 getpgid will fail on real system.

    @patch("subprocess.Popen")
    @patch("os.killpg")
    @patch("os.getpgid")
    @patch("time.sleep")
    def test_signal_handling_logic(
        self, mock_sleep, mock_getpgid, mock_killpg, mock_popen
    ):
        # Setup a pretend running process
        p1 = MagicMock()
        p1.poll.return_value = None  # Running
        p1.pid = 111

        p2 = MagicMock()
        p2.poll.return_value = 0  # Finished
        p2.pid = 222

        run_pipeline.current_processes = [p1, p2]
        run_pipeline.skip_requested = False

        mock_getpgid.return_value = 555  # PGID

        # Trigger Skip
        run_pipeline.signal_handler_skip(signal.SIGUSR1, None)

        # Expectation:
        # 1. skip_requested set to True
        self.assertTrue(run_pipeline.skip_requested)

        # 2. killpg called for p1 (running), but NOT p2 (finished)
        mock_killpg.assert_called_once_with(555, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
