"""
Multimodal benchmark implementations for speculative decoding evaluation.

Each benchmark lives in its own module and registers itself with `MM_BENCHMARKS`,
so that `bench_mmflash.py` can look it up by name:

```python
# mm_benchmarker/mmstar.py
from benchmarker.base import Benchmarker
from benchmarker.utils import create_image_sgl_function

from .registry import MM_BENCHMARKS


@MM_BENCHMARKS.register("mmstar")
class MMStarBenchmarker(Benchmarker):
    ...
```

The module then has to be imported below, otherwise the registration never runs.
"""

from .chartqa import ChartQABenchmarker
from .mathvision import MathVisionBenchmarker
from .mmmu import MMMUBenchmarker
from .mmstar import MMStarBenchmarker
from .ocrbench import OCRBenchBenchmarker
from .realworldqa import RealWorldQABenchmarker
from .registry import MM_BENCHMARKS
from .simplevqa import SimpleVQABenchmarker

__all__ = [
    "MM_BENCHMARKS",
    "ChartQABenchmarker",
    "MathVisionBenchmarker",
    "MMMUBenchmarker",
    "MMStarBenchmarker",
    "OCRBenchBenchmarker",
    "RealWorldQABenchmarker",
    "SimpleVQABenchmarker",
]
