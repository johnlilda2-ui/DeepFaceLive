"""Opt-in allocator reuse for the pinned, standalone offline LHM++ probe.

Only a cloned animation_infer function sees the CUDA proxy. torch.cuda itself,
other model methods, the desktop application and the upstream checkout are
never modified. This adapter is not a shared service concurrency design.
"""
import inspect
from types import FunctionType, MethodType


class _CudaProxy:
    def __init__(self, cuda, stats):
        self._cuda, self._stats = cuda, stats

    def empty_cache(self):
        self._stats['suppressed_empty_cache_calls'] += 1

    def __getattr__(self, name):
        return getattr(self._cuda, name)


class _TorchProxy:
    def __init__(self, torch_module, stats):
        self._torch = torch_module
        self.cuda = _CudaProxy(torch_module.cuda, stats)

    def __getattr__(self, name):
        return getattr(self._torch, name)


def _clone_with_allocator_reuse(method, torch_module, stats):
    original = inspect.unwrap(method.__func__)
    namespace = dict(original.__globals__)
    if namespace.get('torch') is not torch_module:
        raise RuntimeError('Unexpected torch binding in render function')
    namespace['torch'] = _TorchProxy(torch_module, stats)
    clone = FunctionType(original.__code__, namespace, original.__name__,
                         original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    return torch_module.no_grad()(MethodType(clone, method.__self__))


def enable_render_allocator_reuse(model, torch_module):
    if type(model).__module__ != 'core.models.modeling_humana4o_lrm':
        raise RuntimeError('Allocator adapter requires the audited LHM++ class')
    method = model.animation_infer
    original = inspect.unwrap(method.__func__)
    if original.__qualname__ != 'ModelHumanA4OLRM.animation_infer':
        raise RuntimeError('Unexpected animation method')
    if inspect.getsource(original).count('torch.cuda.empty_cache()') != 1:
        raise RuntimeError('Expected exactly one audited per-frame cache release')
    stats = {'scope': 'cloned_animation_infer_only',
             'suppressed_empty_cache_calls': 0,
             'global_torch_cuda_unchanged': True}
    model.animation_infer = _clone_with_allocator_reuse(method, torch_module, stats)
    return stats
