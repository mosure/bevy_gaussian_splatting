"""Check orchestration without Cargo, an adapter, or a renderer."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class QualificationRunnerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash"), "the qualification entrypoint uses bash")
    def test_core_runner_explicitly_selects_both_abis_and_runs_ignored_traversal(self):
        script = Path(__file__).with_name("run_lod_gpu_qualification.sh")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "commands.jsonl"
            cargo = directory / "cargo"
            cargo.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                             "with open(os.environ['BGS_TEST_COMMANDS'],'a') as f:\n"
                             " f.write(json.dumps({'args':sys.argv[1:],'gpu':os.getenv('RUN_GPU_RENDER_TESTS')})+'\\n')\n")
            cargo.chmod(0o755)
            env = os.environ | {"PATH": str(directory) + os.pathsep + os.environ["PATH"],
                                "BGS_TEST_COMMANDS": str(output), "BGS_RUN_GPU_QUALIFICATION": "1",
                                "BGS_GPU_QUALIFICATION_SUITE": "core", "BGS_RUST_TOOLCHAIN": "test-pin"}
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(commands), 4)
            for offset, profile in ((0, "sh0"), (2, "sh3")):
                integration, traversal = commands[offset:offset + 2]
                for command in (integration, traversal):
                    args = command["args"]
                    self.assertEqual(command["gpu"], "1")
                    self.assertEqual(args[0], "+test-pin")
                    self.assertIn("lod_build_" + profile, args[args.index("--features") + 1].split())
                self.assertIn("--include-ignored", integration["args"])
                self.assertIn("gpu_lod_package", integration["args"])
                self.assertIn("near_clip_render", integration["args"])
                self.assertIn("gpu_traversal_matches_cpu_cuts_and_preserves_bounded_parent_fallback", traversal["args"])
                self.assertIn("--ignored", traversal["args"])
            denied = subprocess.run(["bash", str(script)], env=env | {"BGS_RUN_GPU_QUALIFICATION": "0"},
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(denied.returncode, 2)
            self.assertEqual(len(output.read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
