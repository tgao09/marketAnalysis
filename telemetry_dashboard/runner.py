from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any

from telemetry_dashboard.parsers import LineParser


@dataclass
class ActiveProcess:
    command: list[str]
    cwd: Path
    process: subprocess.Popen[str]
    thread: threading.Thread
    parser: LineParser


class ProcessRunner:
    def __init__(self, event_queue: queue.Queue[dict[str, Any]]):
        self.event_queue = event_queue
        self.active: ActiveProcess | None = None

    def start(self, spec_id: str, command: list[str], cwd: Path) -> None:
        parser = LineParser(spec_id)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        worker = threading.Thread(
            target=self._stream_output,
            args=(process, parser),
            daemon=True,
        )
        self.active = ActiveProcess(command=command, cwd=cwd, process=process, thread=worker, parser=parser)
        worker.start()
        self.event_queue.put({"kind": "process_started", "pid": process.pid})

    def _stream_output(self, process: subprocess.Popen[str], parser: LineParser) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            for event in parser.parse(line):
                self.event_queue.put(event)
        returncode = process.wait()
        self.event_queue.put({"kind": "run_finished", "returncode": returncode})

    def stop(self) -> None:
        if self.active is None:
            return
        process = self.active.process
        if process.poll() is None:
            process.terminate()
