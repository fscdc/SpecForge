"""
Multimodal benchmark implementations for speculative decoding evaluation.

Each benchmark lives in its own module and registers itself with `MM_BENCHMARKS`,
so that `bench_mm.py` can look it up by name:

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
from .charxiv import CharXivBenchmarker
from .dynamath import DynaMathBenchmarker
from .longvideobench import LongVideoBenchBenchmarker
from .mathverse import MathVerseBenchmarker
from .mathvision import MathVisionBenchmarker
from .mathvista import MathVistaBenchmarker
from .mmbench import MMBenchBenchmarker
from .mmmu import MMMUBenchmarker
from .mmstar import MMStarBenchmarker
from .moviechat import MovieChatBenchmarker
from .mvbench import MVBenchBenchmarker
from .ocrbench import OCRBenchBenchmarker
from .realworldqa import RealWorldQABenchmarker
from .registry import MM_BENCHMARKS
from .seedbench_image import SEEDBenchImageBenchmarker
from .simplevqa import SimpleVQABenchmarker
from .textvqa import TextVQABenchmarker
from .vdc import VDCBenchmarker
from .videomme import VideoMMEBenchmarker

__all__ = [
    "MM_BENCHMARKS",
    "ChartQABenchmarker",
    "CharXivBenchmarker",
    "DynaMathBenchmarker",
    "LongVideoBenchBenchmarker",
    "MathVerseBenchmarker",
    "MathVisionBenchmarker",
    "MathVistaBenchmarker",
    "MMBenchBenchmarker",
    "MMMUBenchmarker",
    "MMStarBenchmarker",
    "MovieChatBenchmarker",
    "MVBenchBenchmarker",
    "OCRBenchBenchmarker",
    "RealWorldQABenchmarker",
    "SEEDBenchImageBenchmarker",
    "TextVQABenchmarker",
    "SimpleVQABenchmarker",
    "VDCBenchmarker",
    "VideoMMEBenchmarker",
]
