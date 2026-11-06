"""CPU-only regression tests for the GPU verification gate."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import headless_inference_smoke_test as smoke


class EvidenceTests(unittest.TestCase):
    def test_notebook_and_embedded_verification_script_compile(self):
        notebook = json.loads((smoke.REPO_ROOT / "notebooks/kaggle_headless_inference_smoke_test.ipynb").read_text())
        embedded = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "verify_code" for t in node.targets):
                    ast.parse(ast.literal_eval(node.value))
                    embedded += 1
        self.assertEqual(embedded, 1)

    def test_only_executed_compute_nodes_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps([
                {"cat": "Node", "args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}},
                {"cat": "Node", "args": {"provider": "CUDAExecutionProvider", "op_name": "MemcpyFromHost"}},
                {"cat": "Session", "args": {"provider": "CUDAExecutionProvider"}},
            ]))
            self.assertEqual(smoke.profile_counts(path), {"CPUExecutionProvider": 1})

    def test_missing_operation_and_copy_variants_do_not_prove_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps([
                {"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}},
                {"cat": "Node", "args": {"provider": "CUDAExecutionProvider", "op_name": "MemcpyToDevice"}},
            ]))
            self.assertEqual(smoke.profile_counts(path), {})

    def test_session_constructor_is_restored_after_loading_failure(self):
        original = object()
        fake_ort = SimpleNamespace(InferenceSession=original)
        with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                with smoke.profiled_sessions(Path("unused"), True):
                    self.assertIsNot(fake_ort.InferenceSession, original)
                    raise RuntimeError("model load failed")
        self.assertIs(fake_ort.InferenceSession, original)

    def check_report(self, missing_cuda):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = []
            for index in range(4):
                path = root / f"{index}.json"
                provider = "CPUExecutionProvider" if index == missing_cuda else "CUDAExecutionProvider"
                path.write_text(json.dumps([{"cat": "Node", "args": {"provider": provider, "op_name": "Conv"}}]))
                sessions.append(SimpleNamespace(
                    end_profiling=lambda p=str(path): p,
                    get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ))
            models = (SimpleNamespace(_sess=sessions[0]), SimpleNamespace(_sess=sessions[1]),
                      SimpleNamespace(_sess_swap=sessions[2], _sess_rec=sessions[3]))
            with patch.dict("sys.modules", {"onnxruntime": SimpleNamespace(__version__="test")}), \
                 patch.object(smoke.subprocess, "check_output", return_value="commit\n"), \
                 patch.object(smoke.subprocess, "run", return_value=SimpleNamespace(stdout="test GPU")):
                if missing_cuda is None:
                    smoke.save_execution_evidence(models, root, True)
                else:
                    with self.assertRaises(RuntimeError):
                        smoke.save_execution_evidence(models, root, True)
            report = json.loads((root / "execution_evidence.json").read_text())
            self.assertEqual(report["cuda_verified"], missing_cuda is None)
            self.assertEqual(len(report["models"]), 4)

    def test_enabled_cuda_without_execution_fails_for_each_model(self):
        for index in range(4):
            with self.subTest(model=index):
                self.check_report(index)

    def test_all_four_models_executing_cuda_pass(self):
        self.check_report(None)


if __name__ == "__main__":
    unittest.main()
