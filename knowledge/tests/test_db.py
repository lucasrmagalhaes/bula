from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bula_knowledge.db import Database, drop_dead_handles, holds_pgserver


def test_dead_processes_are_dropped_from_the_pgserver_handle_list(tmp_path: Path) -> None:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    handles = tmp_path / ".handle_pids.json"
    handles.write_text(json.dumps([finished.pid, os.getpid()]))

    drop_dead_handles(tmp_path)

    assert json.loads(handles.read_text()) == [os.getpid()]


def test_reused_pid_of_another_program_does_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # O Windows reaproveita PIDs: o número de um serviço morto pode ser hoje de outro
    # programa. Só processo Python conta como dono do Postgres embutido.
    import psutil

    class _OtherProgram:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def name(self) -> str:
            return "python.exe" if self.pid == os.getpid() else "StartMenuExperienceHost.exe"

    monkeypatch.setattr(psutil, "Process", _OtherProgram)
    handles = tmp_path / ".handle_pids.json"
    handles.write_text(json.dumps([12184, os.getpid()]))

    drop_dead_handles(tmp_path)

    assert json.loads(handles.read_text()) == [os.getpid()]
    assert holds_pgserver("12184") is False


def test_missing_or_broken_handle_list_is_left_alone(tmp_path: Path) -> None:
    drop_dead_handles(tmp_path)  # sem arquivo
    (tmp_path / ".handle_pids.json").write_text("não é json")
    drop_dead_handles(tmp_path)
    assert (tmp_path / ".handle_pids.json").read_text() == "não é json"


class _FakeServer:
    def get_uri(self) -> str:
        return "postgresql://fake"


def _bare_database() -> Database:
    database = Database.__new__(Database)
    database._server = None
    return database


def test_embedded_start_waits_out_a_crash_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pgserver

    attempts: list[Path] = []
    naps: list[float] = []

    def slow_then_ready(pgdata: Path, **_: object) -> _FakeServer:
        attempts.append(pgdata)
        if len(attempts) < 3:
            raise subprocess.TimeoutExpired(cmd="pg_ctl start", timeout=10)
        return _FakeServer()

    monkeypatch.setattr(pgserver, "get_server", slow_then_ready)
    monkeypatch.setattr("bula_knowledge.db.time.sleep", naps.append)

    assert _bare_database()._start_embedded(tmp_path) == "postgresql://fake"
    assert len(attempts) == 3
    assert naps == [15, 15]


def test_embedded_start_gives_up_with_a_pointer_to_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pgserver

    def never_ready(pgdata: Path, **_: object) -> _FakeServer:
        raise subprocess.TimeoutExpired(cmd="pg_ctl start", timeout=10)

    monkeypatch.setattr(pgserver, "get_server", never_ready)
    monkeypatch.setattr("bula_knowledge.db.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="não subiu a tempo"):
        _bare_database()._start_embedded(tmp_path)
