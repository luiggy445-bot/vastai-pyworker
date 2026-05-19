import os
import random
import string

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)


def chat_benchmark_generator() -> dict:
    """Generate a lightweight benchmark request (no thinking)."""
    return {
        "model": "qwen36-nudeon",
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 16,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18000,
    model_log_file="/var/log/worker.log",
    handlers=[
        HandlerConfig(
            route="/v1/chat/completions",
            allow_parallel_requests=True,
            max_queue_time=60.0,
            workload_calculator=lambda p: float(p.get("max_tokens", 512)),
            benchmark_config=BenchmarkConfig(
                generator=chat_benchmark_generator,
                runs=1,
                concurrency=1,
            ),
        ),
        HandlerConfig(
            route="/v1/completions",
            allow_parallel_requests=True,
            max_queue_time=60.0,
            workload_calculator=lambda p: float(p.get("max_tokens", 512)),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["all slots are idle"],
        on_error=["CUDA error", "out of memory", "Segmentation fault"],
        on_info=["model loaded"],
    ),
)

Worker(worker_config).run()
