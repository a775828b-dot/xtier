"""Unit test for XTier's CPU multimodal image/video decoding & bounded CPU worker pool (xtier/platform.py).

Verifies without initializing CUDA:
1. Hooked ``media_io.load_image`` decodes real JPEG bytes to PIL RGB on CPU even when callers pass
   ``gpu_image_decode=True`` or ``'nvjpeg_fancy'``, leaving ``torch.cuda.is_initialized() == False``.
2. Hooked ``media_io.load_video`` always forces ``use_gpu=False``.
3. Hooked ``BaseMultimodalProcessor`` methods keep fast image preprocessing on ``'cpu'``, report non-competing
   with the GPU scheduler, set ``gpu_image_decode=False`` / ``video_preprocessing_device='cpu'`` on subclasses
   that override them, preserve ``pickle`` compatibility of ``_load_single_item``, and cap ``SGLANG_CPU_WORKERS``
   + auto processor workers inside XTier's reserved non-MoE CPU headroom (4-8 on large hosts instead of 128).
4. ``XTIER_CPU_IMAGE=0`` opts out of the CPU-decode hooks while keeping the CPU worker budget hook.
"""
import io
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image


class _FakeRecorder:
    def __init__(self):
        self.hooks = {}

    def register(self, target, fn, hook_type):
        self.hooks[target] = (fn, hook_type)


def _make_jpeg_bytes() -> bytes:
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    arr[:, :24, 0] = 200
    arr[:, 24:, 2] = 180
    buf = io.BytesIO()
    Image.fromarray(arr, mode='RGB').save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def main():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from xtier import platform

    # 1. Verify opt-out via XTIER_CPU_IMAGE=0 before applying real hooks.
    old_cpu_img = os.environ.get('XTIER_CPU_IMAGE')
    try:
        os.environ['XTIER_CPU_IMAGE'] = '0'
        rec = _FakeRecorder()
        fixes = platform.install_serving(rec, HookType)
        assert 'cpu-image-decode' not in fixes, f'expected opt-out, got {fixes}'
        assert 'mm-cpu-worker-budget' in fixes, f'expected worker budget fix, got {fixes}'
        assert platform.MM_LOAD_IMAGE not in rec.hooks
        assert platform.MM_PROC_INIT in rec.hooks
    finally:
        if old_cpu_img is None:
            os.environ.pop('XTIER_CPU_IMAGE', None)
        else:
            os.environ['XTIER_CPU_IMAGE'] = old_cpu_img

    # 2. Verify default-on registration and apply hooks to SGLang.
    fixes = platform.install_serving(HookRegistry, HookType)
    assert 'cpu-image-decode' in fixes and 'mm-cpu-worker-budget' in fixes, fixes
    HookRegistry.apply_hooks()

    from sglang.srt.multimodal import media_io
    from sglang.srt.multimodal.modality import Modality
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    # 3. Real JPEG decoding stays on CPU even with gpu_image_decode=True / 'nvjpeg_fancy'.
    jpg = _make_jpeg_bytes()
    ref = Image.open(io.BytesIO(jpg)).convert('RGB')
    for mode in (True, 'nvjpeg_fancy'):
        img1, _ = media_io.load_image(jpg, mode)
        img2, _ = media_io.load_image(jpg, gpu_image_decode=mode)
        for img in (img1, img2):
            assert isinstance(img, Image.Image), f'expected PIL Image, got {type(img)}'
            assert np.array_equal(np.asarray(img.convert('RGB')), np.asarray(ref)), 'pixel mismatch'
    assert torch.cuda.memory_allocated() == 0 and torch.cuda.memory_reserved() == 0, (
        f'GPU memory allocated during CPU image decode: alloc={torch.cuda.memory_allocated()} res={torch.cuda.memory_reserved()}'
    )

    # 4. load_video forces use_gpu=False.
    seen_use_gpu = []

    def _dummy_load_video(video_file, use_gpu=True):
        seen_use_gpu.append(use_gpu)
        return video_file

    platform._mm_load_video_cpu(_dummy_load_video, 'fake.mp4', True)
    platform._mm_load_video_cpu(_dummy_load_video, 'fake.mp4', use_gpu=True)
    platform._mm_load_video_cpu(_dummy_load_video, 'fake.mp4')
    assert seen_use_gpu == [False, False, False], f'load_video use_gpu not forced False: {seen_use_gpu}'

    # 5. Subclass overriding gpu_image_decode / video_preprocessing_device stays CPU + picklable.
    class _DummyVLProcessor(BaseMultimodalProcessor):
        gpu_image_decode = 'nvjpeg_fancy'
        video_preprocessing_device = None

        async def process_mm_data_async(self, *a, **kw):
            return None

    proc = object.__new__(_DummyVLProcessor)
    # Call hooked _fast_image_processor_device & _preprocessing_competes_with_the_scheduler.
    assert proc._fast_image_processor_device(None) == 'cpu'
    assert proc._preprocessing_competes_with_the_scheduler() is False

    # Call _mm_processor_init wrapper directly with a lightweight dummy original.
    old_sglang_cpu = os.environ.pop('SGLANG_CPU_WORKERS', None)
    old_mm_workers = os.environ.pop('XTIER_MM_CPU_WORKERS', None)
    try:
        seen_env = []
        platform._mm_processor_init(lambda self: seen_env.append(os.environ.get('SGLANG_CPU_WORKERS')), proc)
        assert _DummyVLProcessor.gpu_image_decode is False
        assert proc.gpu_image_decode is False
        assert _DummyVLProcessor.video_preprocessing_device == 'cpu'
        assert proc.video_preprocessing_device == 'cpu'
        assert seen_env and seen_env[0] is not None
        auto_budget = int(seen_env[0])
        assert 1 <= auto_budget <= 8, f'expected bounded worker budget 1..8, got {auto_budget}'

        # _load_single_item remains picklable and decodes on CPU.
        item = _DummyVLProcessor._load_single_item(jpg, Modality.IMAGE)
        assert isinstance(item, Image.Image)
        assert np.array_equal(np.asarray(item), np.asarray(ref))
        assert torch.cuda.memory_allocated() == 0 and torch.cuda.memory_reserved() == 0
        assert pickle.dumps(BaseMultimodalProcessor._load_single_item)

        # _mm_auto_workers boosts 2 -> min(4, budget) and clamps excessive values to budget.
        assert platform._mm_auto_workers(lambda self: 2, proc) == min(4, auto_budget)
        assert platform._mm_auto_workers(lambda self: 64, proc) == auto_budget

        # Explicit XTIER_MM_CPU_WORKERS override wins.
        os.environ['XTIER_MM_CPU_WORKERS'] = '5'
        assert platform._mm_cpu_budget() == 5
        assert platform._mm_auto_workers(lambda self: 2, proc) == 5

        # Explicit SGLANG_CPU_WORKERS is preserved (setdefault).
        os.environ['SGLANG_CPU_WORKERS'] = '3'
        seen_env.clear()
        platform._mm_processor_init(lambda self: seen_env.append(os.environ.get('SGLANG_CPU_WORKERS')), proc)
        assert seen_env == ['3']
    finally:
        if old_sglang_cpu is None:
            os.environ.pop('SGLANG_CPU_WORKERS', None)
        else:
            os.environ['SGLANG_CPU_WORKERS'] = old_sglang_cpu
        if old_mm_workers is None:
            os.environ.pop('XTIER_MM_CPU_WORKERS', None)
        else:
            os.environ['XTIER_MM_CPU_WORKERS'] = old_mm_workers

    print(f'PASS cpu_image_check: auto_budget={auto_budget}, cuda_alloc={torch.cuda.memory_allocated()}, cuda_res={torch.cuda.memory_reserved()}')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print(f'FAIL cpu_image_check: {e}', file=sys.stderr)
        sys.exit(1)
