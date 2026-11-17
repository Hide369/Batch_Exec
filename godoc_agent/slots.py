"""サーバー全体で同時に動かす Claude Code のセッション数の上限（合計 MAX_CONCURRENCY = 10）。

Web 画面から複数人が別々の出力先で同時に実行したり、コマンドからの実行と重なったりしても、
推論サーバー（vLLM）に同時に送るセッションの合計をここで抑える。runner はそれぞれ別プロセスで動くため、
プロセスをまたいで数えられるよう、枠ごとのロックファイル（<state>/llm_slots/slot-01.lock …）を
OS のファイルロックで取り合う。プロセスが終了すれば（強制終了でも）OS がロックを解放するので、枠が漏れることはない。
"""

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

from .config import APP_DIR, MAX_CONCURRENCY
from .runlock import try_lock_file, unlock_file

DEFAULT_SLOTS_DIR = APP_DIR / "state" / "llm_slots"
# 全枠が埋まっているとき、空きを確かめ直す間隔（秒）。1回の Claude Code の実行は数分かかるため十分に短い
POLL_INTERVAL_S = 1.0
# 画面表示用の使用数を使い回す秒数。複数の画面が同時に問い合わせても、同じ数を返し、枠を確かめに行く回数を抑える
BUSY_COUNT_CACHE_S = 1.0


class GlobalSlot:
    """取得したサーバー全体の枠1つ。release() で返す。"""

    def __init__(self, number: int, f: IO[str]):
        self.number = number
        self._file: IO[str] | None = f

    def release(self) -> None:
        if self._file is None:
            return
        try:
            unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None


class GlobalSlots:
    def __init__(
        self, directory: Path | None = None, limit: int = MAX_CONCURRENCY, poll_interval_s: float = POLL_INTERVAL_S,
        *, busy_cache_s: float = 0.0, clock: Callable[[], float] = time.monotonic,
    ):
        if limit < 1:
            raise ValueError(f"同時セッション数の上限は1以上で指定してください: {limit}")
        # 既定値は呼び出し時に決める（テストで置き場所を差し替えられるように）
        self.directory = directory or DEFAULT_SLOTS_DIR
        self.limit = limit
        self.poll_interval_s = poll_interval_s
        self.busy_cache_s = busy_cache_s
        self._clock = clock
        # busy_count の確かめ方（空き枠を一瞬だけ取る）は、同じプロセスの別スレッドが同時に行うと、
        # 互いに相手が取った枠を「使用中」と数えてしまう（Web 画面を複数のブラウザで開くと、画面ごとに数が食い違う）。
        # そのため確かめるのは同時に1つだけにする
        self._probe_lock = threading.Lock()
        self._cached: tuple[float, int] | None = None

    def _path(self, number: int) -> Path:
        return self.directory / f"slot-{number:02d}.lock"

    def try_acquire(self) -> GlobalSlot | None:
        """空いている枠を1つ取る。すべて使用中なら None。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        for number in range(1, self.limit + 1):
            f = self._path(number).open("a+", encoding="utf-8")
            if try_lock_file(f):
                return GlobalSlot(number, f)
            f.close()
        return None

    async def acquire(self, on_wait: Callable[[], None] | None = None) -> GlobalSlot:
        """空いている枠を1つ取る。すべて使用中なら空くまで待つ（待ち始めに一度だけ on_wait を呼ぶ）。"""
        waited = False
        while (slot := self.try_acquire()) is None:
            if not waited and on_wait is not None:
                on_wait()
            waited = True
            await asyncio.sleep(self.poll_interval_s)
        return slot

    def busy_count(self) -> int:
        """いま使用中の枠の数（画面表示用）。

        確かめる一瞬だけ空き枠を取るため、取り合う側は次の確認で取り直す。同じプロセスの中では同時に1つだけ確かめ、
        busy_cache_s 秒以内の問い合わせには直前の結果を返す（どのブラウザの画面にも同じ数が出るように）。
        """
        with self._probe_lock:
            now = self._clock()
            if self._cached is not None and now - self._cached[0] < self.busy_cache_s:
                return self._cached[1]
            busy = self._probe()
            self._cached = (self._clock(), busy)
            return busy

    def _probe(self) -> int:
        if not self.directory.is_dir():
            return 0
        busy = 0
        for number in range(1, self.limit + 1):
            path = self._path(number)
            if not path.is_file():
                continue
            with path.open("a+", encoding="utf-8") as f:
                if try_lock_file(f):
                    unlock_file(f)
                else:
                    busy += 1
        return busy
