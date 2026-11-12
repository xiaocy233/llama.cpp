import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("moe_adaptive", Path(__file__).resolve().parents[1] / "scripts" / "moe_adaptive.py")
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        model, server = root / "model.gguf", root / "server.exe"
        model.touch()
        server.touch()
        self.config = {
            "version": 1,
            "command": [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", "8080",
                        "--log-file", "production.log", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                        "--spec-draft-p-min", "0", "--reasoning", "on", "-c", "10240", "-b", "2048", "-ub", "2048",
                        "--spec-mtp-ubatch", "128", "--n-cache-slots", "86", "--n-cache-predict", "8"],
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "GGML_MOE_HYBRID_POLICY": "legacy",
                            "GGML_MOE_HYBRID": "0", "GGML_MOE_HYBRID_MAX_EXPERTS": "0",
                            "GGML_MOE_PREFETCH_BUDGET": "4", "GGML_MOE_CACHE_POLICY": "decay",
                            "GGML_MOE_CACHE_DECAY": "8", "GGML_MOE_LAYER_SLOTS": "40,60",
                            "LLAMA_COMPACT_GREEDY": "1", "GGML_MOE_BATCH_COPY": "1"},
            "main_threads": 1, "expert_threads": 3, "cpu_set": "0-3",
            "selected": {"prefetch_budget": 4, "threshold": 0, "cpu_cap": 0}}
        self.args = argparse.Namespace(launcher=root / "start-server.ps1", gpu=None, thinking=None,
                                      action="tune", profile=None, output=root, window=32, cost_rounds=48,
                                      trial_budget=24, warmup_tokens=0, tokens=512, confirm_tokens=1024)
        with patch.object(adaptive.subprocess, "run") as run:
            run.return_value.stdout = json.dumps(self.config)
            self.args.deployment = adaptive.deployment(self.args)

    def test_ordinary_server_keeps_prediction_width_and_no_instrumentation(self):
        for cpu in (0, 4):
            config = {"prefetch_budget": 3, "threshold": 0.6, "cpu_cap": cpu}
            command, env = adaptive.production_command(self.args, 4567, config, "check.log")
            self.assertEqual(command[command.index("--n-cache-predict") + 1], "8")
            self.assertEqual(command[command.index("--spec-draft-p-min") + 1], "0.6")
            self.assertEqual(env["GGML_MOE_PREFETCH_BUDGET"], "3")
            self.assertEqual(env["GGML_MOE_HYBRID"], str(int(cpu > 0)))
            self.assertFalse(any(k.startswith("LLAMA_MOE_AUTOTUNE") for k in env))
        self.assertEqual(self.config["environment"]["GGML_MOE_PREFETCH_BUDGET"], "4")

    def test_native_collection_preserves_production_options(self):
        command, env, manifest = adaptive.launch(self.args, 4567)
        for option, expected in (("-b", "2048"), ("-ub", "2048"), ("-c", "10240"),
                                 ("--spec-mtp-ubatch", "128"), ("--n-cache-predict", "8")):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(env["GGML_MOE_CACHE_POLICY"], "decay")
        self.assertEqual(env["GGML_MOE_BATCH_COPY"], "1")
        self.assertEqual(env["LLAMA_MOE_AUTOTUNE_PREFETCH"], "4")
        self.assertEqual(manifest["production"], self.config)

    def test_cost_identity_changes_with_deployment(self):
        _, first, _ = adaptive.launch(self.args, 4567)
        self.args.deployment["environment"]["GGML_MOE_LAYER_SLOTS"] = "50,50"
        _, second, _ = adaptive.launch(self.args, 4567)
        self.assertNotEqual(first["LLAMA_MOE_AUTOTUNE_PROFILE_KEY"], second["LLAMA_MOE_AUTOTUNE_PROFILE_KEY"])

    def test_inherited_tuner_state_rejected(self):
        self.config["environment"]["LLAMA_MOE_AUTOTUNE"] = "online"
        with patch.object(adaptive.subprocess, "run") as run:
            run.return_value.stdout = json.dumps(self.config)
            with self.assertRaisesRegex(ValueError, "Remove LLAMA_MOE_AUTOTUNE"):
                adaptive.deployment(self.args)

    def test_decode_excludes_prefill_output_token(self):
        self.assertEqual(adaptive.decode_token_count({"predicted_n": 1024, "decode_output_tokens": 1023}), 1023)


if __name__ == "__main__":
    unittest.main()
