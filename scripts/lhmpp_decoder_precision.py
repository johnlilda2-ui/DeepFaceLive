"""Opt-in decoder-only CUDA FP16 autocast for the standalone offline probe."""
from functools import wraps


def enable_decoder_fp16(model, torch):
    decoder = model.neural_renderer
    if (type(decoder).__module__ != 'core.models.transformer_block.dpt_decoder'
            or type(decoder).__name__ != 'PatchDPT4DecoderOnly'):
        raise RuntimeError('Requires the audited PatchDPT4DecoderOnly')
    if not all(p.is_cuda and p.dtype == torch.float32 for p in decoder.parameters()):
        raise RuntimeError('Decoder weights must start as CUDA float32')
    if getattr(decoder, '_dfl_fp16_probe', False):
        raise RuntimeError('Decoder autocast is already enabled')
    original = decoder.forward
    stats = {'scope': 'neural_decoder_only', 'autocast_dtype': 'float16',
             'weight_dtype': 'float32', 'calls': 0, 'observed_output_dtypes': [],
             'finite_outputs_checked': True}

    @wraps(original)
    def forward(*args, **kwargs):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            values = original(*args, **kwargs)
        if not isinstance(values, tuple) or len(values) != 2:
            raise RuntimeError('Unexpected decoder output structure')
        result = []
        for value in values:
            if value is None:
                result.append(None)
                continue
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                raise RuntimeError('Decoder output must remain on CUDA')
            if not torch.isfinite(value).all().item():
                raise RuntimeError('Nonfinite decoder output before uint8 encoding')
            dtype = str(value.dtype)
            if dtype not in stats['observed_output_dtypes']:
                stats['observed_output_dtypes'].append(dtype)
            # Keep compositing and uint8 conversion outside autocast.
            result.append(value.float())
        stats['calls'] += 1
        return tuple(result)

    decoder.forward = forward
    decoder._dfl_fp16_probe = True
    return stats
