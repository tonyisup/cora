"""Encrypted, owner-controlled storage for authorized private audio sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from typing import Callable


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_SESSION_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,79}")


def _run(
    runner: CommandRunner,
    command: list[str],
    *,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        command,
        input=stdin,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"command failed ({command[0]}): {detail}")
    return result


def create_encrypted_session_storage(
    *,
    session_id: str,
    base_dir: str | Path,
    mount_root: str | Path,
    runner: CommandRunner = subprocess.run,
    password_factory: Callable[[], str] = lambda: secrets.token_urlsafe(48),
) -> dict[str, object]:
    """Create and mount a new AES-256 APFS sparse bundle for one session."""

    if _SESSION_ID.fullmatch(session_id) is None:
        raise ValueError("session_id must be 8-80 lowercase letters, digits, or hyphens")
    if shutil.which("hdiutil") is None or shutil.which("tmutil") is None:
        raise RuntimeError("encrypted session storage requires hdiutil and tmutil")
    base = Path(base_dir).expanduser().resolve()
    mounts = Path(mount_root).resolve()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(base, 0o700)
    image = base / f"{session_id}.sparsebundle"
    key_path = base / f".{session_id}.key"
    mountpoint = mounts / session_id
    if image.exists() or key_path.exists() or mountpoint.exists():
        raise ValueError("encrypted session storage already exists")

    password = password_factory()
    if len(password) < 32:
        raise ValueError("per-session encryption password is too short")
    descriptor = os.open(
        key_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    exclusion_modes: dict[str, str] = {}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(password + "\n")
        os.chmod(key_path, 0o600)
        _run(
            runner,
            [
                "hdiutil",
                "create",
                "-size",
                "256m",
                "-type",
                "SPARSEBUNDLE",
                "-fs",
                "APFS",
                "-volname",
                f"Cora-{session_id[-12:]}",
                "-encryption",
                "AES-256",
                "-stdinpass",
                str(image),
            ],
            stdin=password + "\n",
        )
        mountpoint.mkdir(parents=True, mode=0o700)
        _run(
            runner,
            [
                "hdiutil",
                "attach",
                "-nobrowse",
                "-mountpoint",
                str(mountpoint),
                "-stdinpass",
                str(image),
            ],
            stdin=password + "\n",
        )
        for protected_path in (image, key_path):
            fixed = runner(
                ["tmutil", "addexclusion", "-p", str(protected_path)],
                capture_output=True,
                check=False,
                text=True,
            )
            if fixed.returncode == 0:
                exclusion_modes[str(protected_path)] = "fixed_path"
            else:
                # macOS requires root for -p. The default sticky exclusion is
                # usable by the owner and follows the file if it is moved.
                _run(runner, ["tmutil", "addexclusion", str(protected_path)])
                exclusion_modes[str(protected_path)] = "sticky_file"
            exclusion = _run(runner, ["tmutil", "isexcluded", str(protected_path)])
            if "[Excluded]" not in exclusion.stdout:
                raise RuntimeError(f"backup exclusion did not verify: {protected_path}")
    except Exception:
        if mountpoint.exists():
            runner(
                ["hdiutil", "detach", str(mountpoint), "-force"],
                capture_output=True,
                check=False,
                text=True,
            )
        if key_path.exists():
            key_path.unlink()
        if image.is_dir():
            shutil.rmtree(image)
        elif image.exists():
            image.unlink()
        if mountpoint.exists():
            try:
                mountpoint.rmdir()
            except OSError:
                # Never recursively delete an unexpectedly non-empty mount.
                pass
        raise

    session_root = mountpoint / "session"
    session_root.mkdir(mode=0o700)
    record = {
        "schema_version": 1,
        "session_id": session_id,
        "storage_alias": f"cora-private:{session_id}",
        "image_path": str(image),
        "mountpoint": str(mountpoint),
        "session_root": str(session_root),
        "encryption": "AES-256 encrypted APFS sparse bundle",
        "per_session_key": True,
        "key_file_mode": oct(key_path.stat().st_mode & 0o777),
        "key_material_recorded_in_report": False,
        "backup_excluded": True,
        "backup_exclusion_modes": exclusion_modes,
        "retention_days": 30,
        "cryptographic_erasure_method": "delete per-session key file",
    }
    (session_root / "private-storage-record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def detach_encrypted_session_storage(
    storage_record: dict[str, object],
    *,
    runner: CommandRunner = subprocess.run,
) -> None:
    """Unmount a retained encrypted session without deleting its key or image."""

    mountpoint = str(storage_record.get("mountpoint", ""))
    if not mountpoint:
        raise ValueError("storage record is missing mountpoint")
    _run(runner, ["hdiutil", "detach", mountpoint])
