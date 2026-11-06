import os
from pathlib import Path
import tempfile
import unittest

from prepare_headless_env import ensure_environment


class EnvironmentTests(unittest.TestCase):
    def test_repairs_empty_interpreter_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'environment'
            interpreter = ensure_environment(root)
            self.assertFalse(interpreter.is_symlink())
            marker = root / 'lib' / 'existing-package-marker'
            marker.write_text('keep')
            interpreter.unlink()
            interpreter.write_bytes(b'')
            repaired = ensure_environment(root)
            self.assertTrue(os.access(repaired, os.X_OK))
            self.assertGreater(repaired.stat().st_size, 0)
            self.assertEqual(marker.read_text(), 'keep')

    def test_does_not_overwrite_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'user-file'
            marker.write_text('keep')
            with self.assertRaises(ValueError):
                ensure_environment(temp)
            self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
