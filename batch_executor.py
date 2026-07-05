"""批处理任务的串行/并行执行工具。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BatchTaskResult:
    """单个案例任务的执行结果。"""

    case_id: str
    item_index: int
    value: Any = None
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        return self.error is None


def execute_case_tasks(
    tasks: Iterable[Tuple[str, int]],
    worker: Callable[[str, int], Any],
    parallel: bool = False,
    max_workers: int = 4,
) -> List[BatchTaskResult]:
    """
    按输入顺序返回案例任务结果，单个任务异常不会中断其他任务。

    Args:
        tasks: ``(case_id, item_index)`` 任务序列。
        worker: 处理单个任务的函数。
        parallel: 是否启用线程池并行。
        max_workers: 最大并发任务数，必须大于等于 1。
    """
    if max_workers < 1:
        raise ValueError("BATCH_MAX_WORKERS 必须大于等于 1")

    task_list = list(tasks)
    if not task_list:
        return []

    def run_safely(case_id: str, item_index: int) -> BatchTaskResult:
        try:
            return BatchTaskResult(
                case_id=case_id,
                item_index=item_index,
                value=worker(case_id, item_index),
            )
        except Exception as exc:
            return BatchTaskResult(
                case_id=case_id,
                item_index=item_index,
                error=exc,
            )

    if not parallel or len(task_list) == 1:
        return [run_safely(case_id, item_index) for case_id, item_index in task_list]

    ordered_results: List[Optional[BatchTaskResult]] = [None] * len(task_list)
    worker_count = min(max_workers, len(task_list))

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="batch-case",
    ) as executor:
        future_to_index = {
            executor.submit(run_safely, case_id, item_index): index
            for index, (case_id, item_index) in enumerate(task_list)
        }
        for future in as_completed(future_to_index):
            ordered_results[future_to_index[future]] = future.result()

    return [result for result in ordered_results if result is not None]
