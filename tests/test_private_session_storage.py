from __future__ import annotations

import os
import subprocess

import pytest

from cora_infans.private_session_storage import create_encrypted_session_storage


def test_encrypted_storage_uses_stdin_key_and_verifies_backup_exclusion(
    tmp_path, monkeypatch
) -> None:
    commands: list[tuple[list[str], str | None]] = []

    def runner(command, *, input=None, **_kwargs):
        commands.append((list(command), input))
        stdout = "[Excluded] test" if command[1] == "isexcluded" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "cora_infans.private_session_storage.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    record = create_encrypted_session_storage(
        session_id="bvips-v1-20260716-s01",
        base_dir=tmp_path / "private",
        mount_root=tmp_path / "mounts",
        runner=runner,
        password_factory=lambda: "p" * 48,
    )

    create = next(row for row in commands if row[0][1] == "create")
    attach = next(row for row in commands if row[0][1] == "attach")
    assert "p" * 48 not in create[0]
    assert "p" * 48 not in attach[0]
    assert create[1] == "p" * 48 + "\n"
    assert attach[1] == "p" * 48 + "\n"
    assert record["encryption"] == "AES-256 encrypted APFS sparse bundle"
    assert record["per_session_key"] is True
    assert record["backup_excluded"] is True
    assert record["key_file_mode"] == "0o600"
    key_path = tmp_path / "private/.bvips-v1-20260716-s01.key"
    assert os.stat(key_path).st_mode & 0o777 == 0o600


def test_encrypted_storage_fails_closed_when_backup_exclusion_does_not_verify(
    tmp_path, monkeypatch
) -> None:
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="[Included] test", stderr="")

    monkeypatch.setattr(
        "cora_infans.private_session_storage.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    with pytest.raises(RuntimeError, match="backup exclusion did not verify"):
        create_encrypted_session_storage(
            session_id="bvips-v1-20260716-s01",
            base_dir=tmp_path / "private",
            mount_root=tmp_path / "mounts",
            runner=runner,
            password_factory=lambda: "p" * 48,
        )

    assert not (tmp_path / "private/.bvips-v1-20260716-s01.key").exists()
    assert not (tmp_path / "mounts/bvips-v1-20260716-s01").exists()


def test_encrypted_storage_falls_back_to_verified_sticky_exclusion(
    tmp_path, monkeypatch
) -> None:
    commands: list[list[str]] = []

    def runner(command, **_kwargs):
        commands.append(list(command))
        if command[1:3] == ["addexclusion", "-p"]:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="requires root privileges"
            )
        stdout = "[Excluded] test" if command[1] == "isexcluded" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "cora_infans.private_session_storage.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    record = create_encrypted_session_storage(
        session_id="bvips-v1-20260716-s01",
        base_dir=tmp_path / "private",
        mount_root=tmp_path / "mounts",
        runner=runner,
        password_factory=lambda: "p" * 48,
    )

    assert set(record["backup_exclusion_modes"].values()) == {"sticky_file"}
    assert sum(command[1:3] == ["addexclusion", "-p"] for command in commands) == 2
    assert sum(command[1] == "addexclusion" and "-p" not in command for command in commands) == 2


def test_failed_storage_creation_removes_partial_sparse_bundle(tmp_path, monkeypatch) -> None:
    image = tmp_path / "private/bvips-v1-20260716-s01.sparsebundle"

    def runner(command, **_kwargs):
        if command[1] == "create":
            image.mkdir(parents=True)
        if command[1] == "isexcluded":
            return subprocess.CompletedProcess(command, 0, stdout="[Included]", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "cora_infans.private_session_storage.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    with pytest.raises(RuntimeError, match="backup exclusion did not verify"):
        create_encrypted_session_storage(
            session_id="bvips-v1-20260716-s01",
            base_dir=tmp_path / "private",
            mount_root=tmp_path / "mounts",
            runner=runner,
            password_factory=lambda: "p" * 48,
        )
    assert not image.exists()
    assert not (tmp_path / "mounts/bvips-v1-20260716-s01").exists()
