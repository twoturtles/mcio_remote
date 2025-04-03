import time
from queue import Empty
from threading import Thread
from typing import TypeVar

import pytest

from mcio_remote.util import LatestItemQueue

T = TypeVar("T")


def test_latest_item_queue_basic() -> None:
    q: LatestItemQueue[int] = LatestItemQueue()
    assert not q.put(1)  # First item, nothing dropped
    assert q.put(2)  # Second item drops first
    assert q.get() == 2  # Should get latest item


def test_queue_empty_behavior() -> None:
    q: LatestItemQueue[str] = LatestItemQueue()
    with pytest.raises(Empty):
        q.get(block=False)


def test_threaded_operation() -> None:
    q: LatestItemQueue[int] = LatestItemQueue()

    def producer() -> None:
        for i in range(3):
            q.put(i)
            time.sleep(0.01)

    thread = Thread(target=producer)
    thread.start()

    # Wait and get final value
    time.sleep(0.1)
    assert q.get() == 2
    thread.join()
