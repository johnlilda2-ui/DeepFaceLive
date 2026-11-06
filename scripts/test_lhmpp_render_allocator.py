import unittest
from types import SimpleNamespace
from lhmpp_render_allocator import _clone_with_allocator_reuse

torch = None

class Render:
    def frame(self, value=3, *, fail=False):
        torch.cuda.empty_cache()
        if fail:
            raise ValueError('test failure')
        return torch.identity(value)

class AllocatorTests(unittest.TestCase):
    def setUp(self):
        global torch
        self.released = []
        torch = SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: self.released.append(1)),
            identity=lambda x: x, no_grad=lambda: lambda fn: fn)
        self.model = Render()
        self.stats = {'suppressed_empty_cache_calls': 0}
        self.clone = _clone_with_allocator_reuse(self.model.frame, torch, self.stats)

    def test_suppresses_only_clone_and_preserves_result_defaults(self):
        self.assertEqual(self.clone(), 3)
        self.assertEqual(self.stats['suppressed_empty_cache_calls'], 1)
        self.assertEqual(self.released, [])
        self.assertEqual(self.model.frame(7), 7)
        self.assertEqual(self.released, [1])

    def test_failure_does_not_change_global_cuda_or_original_method(self):
        original_cuda = torch.cuda
        original_release = torch.cuda.empty_cache
        with self.assertRaises(ValueError):
            self.clone(fail=True)
        self.assertIs(torch.cuda, original_cuda)
        self.assertIs(torch.cuda.empty_cache, original_release)
        self.model.frame()
        self.assertEqual(self.released, [1])

    def test_unexpected_global_binding_rejected(self):
        with self.assertRaises(RuntimeError):
            _clone_with_allocator_reuse(self.model.frame, object(), self.stats)

if __name__ == '__main__':
    unittest.main()
