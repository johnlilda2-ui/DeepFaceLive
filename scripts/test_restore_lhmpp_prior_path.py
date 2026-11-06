"""CPU regression checks for preserved Kaggle prior-path restoration."""
import tempfile
import unittest
from pathlib import Path
from restore_lhmpp_kaggle_runtime import ensure_prior_link


class PriorPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / 'assets'
        self.target.mkdir()
        (self.target / 'weights').write_bytes(b'verified')
        self.link = self.root / 'pretrained_models'

    def test_absent_creates_expected_link(self):
        ensure_prior_link(self.link, self.target)
        self.assertTrue(self.link.is_symlink())
        self.assertEqual((self.link / 'weights').read_bytes(), b'verified')

    def test_matching_link_is_reused(self):
        self.link.symlink_to(self.target, target_is_directory=True)
        inode = self.link.lstat().st_ino
        ensure_prior_link(self.link, self.target)
        self.assertEqual(self.link.lstat().st_ino, inode)

    def test_empty_restored_entry_is_preserved(self):
        self.link.touch()
        ensure_prior_link(self.link, self.target)
        self.assertEqual(self.link.resolve(), self.target)
        self.assertEqual((self.root / 'pretrained_models.kaggle-empty-backup').read_bytes(), b'')

    def test_unrecognized_entries_are_unchanged(self):
        for kind in ('file', 'directory', 'symlink'):
            with self.subTest(kind=kind):
                link = self.root / kind
                if kind == 'file':
                    link.write_bytes(b'keep')
                elif kind == 'directory':
                    link.mkdir()
                else:
                    link.symlink_to(self.root / 'missing')
                with self.assertRaises(RuntimeError):
                    ensure_prior_link(link, self.target)
                if kind == 'file':
                    self.assertEqual(link.read_bytes(), b'keep')
                elif kind == 'directory':
                    self.assertTrue(link.is_dir())
                else:
                    self.assertTrue(link.is_symlink())

    def test_existing_backup_is_never_overwritten(self):
        self.link.touch()
        backup = self.root / 'pretrained_models.kaggle-empty-backup'
        backup.write_bytes(b'keep')
        with self.assertRaises(RuntimeError):
            ensure_prior_link(self.link, self.target)
        self.assertFalse(self.link.is_symlink())
        self.assertEqual(backup.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
