class MMBenchmarkRegistry:
    """
    Registry for multimodal benchmarks.

    Kept separate from `benchmarker.registry.BENCHMARKS` so that the multimodal
    benchmarks can be listed and looked up independently of the text-only ones.
    """

    def __init__(self):
        self.benchmarks = {}

    def register(self, name: str):
        """
        Usage:
        ```python
            MM_BENCHMARKS = MMBenchmarkRegistry()

            @MM_BENCHMARKS.register("mmstar")
            class MMStarBenchmarker(Benchmarker):
                ...
        ```
        """

        def wrapper(cls):
            self.benchmarks[name] = cls
            return cls

        return wrapper

    def get(self, name: str) -> type:
        """
        Get the multimodal benchmark class by name.
        """
        if name not in self.benchmarks:
            available = ", ".join(sorted(self.benchmarks)) or "<none registered yet>"
            raise KeyError(
                f"Unknown multimodal benchmark '{name}'. Available: {available}"
            )
        return self.benchmarks[name]


MM_BENCHMARKS = MMBenchmarkRegistry()
