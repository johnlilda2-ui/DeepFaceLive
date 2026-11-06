"""Opt-in, inference-only attention adapter for the pinned LHM++ T4 probe.

Preserves patch grouping, order, scale and weights. Replaces the dense K x K
attention allocation with xFormers CUTLASS attention. No FlashAttention-2.
"""
from types import MethodType


def enable_t4_attention(model):
    import torch
    import xformers.ops as xops
    from core.models.encoders.sonata.model import SerializedAttention, offset2bincount

    if model.training:
        raise RuntimeError('This attention adapter supports inference only')
    # Verify the installed operator numerically on the actual GPU before use.
    with torch.inference_mode():
        gen = torch.Generator(device='cuda').manual_seed(42)
        q, k, v = [torch.randn(2, 32, 4, 16, device='cuda', generator=gen)
                   for _ in range(3)]
        scale = 16 ** -0.5
        expected = ((q.transpose(1, 2) * scale) @ k.transpose(1, 2).transpose(-2, -1))
        expected = (expected.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)
        actual = xops.memory_efficient_attention(q, k, v, p=0.0, scale=scale,
                                                op=xops.MemoryEfficientAttentionCutlassOp)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        max_error = (actual - expected).abs().max().item()

    def forward(self, point):
        if self.training:
            raise RuntimeError('Training is forbidden for this adapter')
        self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)
        pad, unpad, _ = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]
        q, k, v = qkv.reshape(-1, self.patch_size, 3, self.num_heads,
                              self.channels // self.num_heads).unbind(dim=2)
        feat = xops.memory_efficient_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), p=0.0,
            scale=self.scale, op=xops.MemoryEfficientAttentionCutlassOp)
        feat = feat.reshape(-1, self.channels)[inverse]
        point.feat = self.proj_drop(self.proj(feat))
        return point

    modules = [m for m in model.modules() if type(m) is SerializedAttention]
    if not modules:
        raise RuntimeError('No audited SerializedAttention modules found')
    for module in modules:
        if (module.enable_flash or module.enable_rpe or module.upcast_attention
                or module.upcast_softmax or module.attn_drop.p != 0):
            raise RuntimeError('Unsupported attention settings; refusing to change semantics')
    for module in modules:
        module.forward = MethodType(forward, module)
    return {'backend': 'xformers_cutlass', 'module_count': len(modules),
            'gpu_numerical_max_abs_error': max_error, 'weights_changed': False}
