"""Persistent subagents with private home-directory overlay filesystems.

This module owns the overlay runtime, submitted artifact materialization,
deterministic diffs, and conflict-checked application.
"""

from __future__ import annotations

import difflib
import io
import os
import stat
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Literal, Mapping, Optional

Operation = Literal["create", "modify", "delete"]

DEFAULT_PER_FILE_LIMIT = 8 * 1024 * 1024
DEFAULT_AGGREGATE_LIMIT = 32 * 1024 * 1024
DEFAULT_SNAPSHOT_BYTE_LIMIT = 1024 * 1024 * 1024
DEFAULT_SNAPSHOT_INODE_LIMIT = 250_000
DEFAULT_SNAPSHOT_TIMEOUT = 120.0
DEFAULT_SNAPSHOT_MIN_FREE_BYTES = 512 * 1024 * 1024


class OverlaySubagentError(Exception):
    """Base error for overlay subagent operations."""


class PathValidationError(OverlaySubagentError):
    """Raised when a submitted path fails validation."""


class ApplyConflict(OverlaySubagentError):
    """Raised when apply cannot safely write because destination state diverged."""

    def __init__(self, path: str, reason: str, conflicts: Optional[list["ApplyConflict"]] = None):
        self.path = path
        self.reason = reason
        self.conflicts = conflicts or []
        if self.conflicts:
            details = "; ".join(f"{c.path}: {c.reason}" for c in self.conflicts)
            message = f"Apply conflicts: {details}"
        else:
            message = f"{path}: {reason}"
        super().__init__(message)


def normalize_submitted_path(path: str) -> str:
    """Normalize a worker-relative submitted path.

    Rules:
    - must be relative
    - must not contain parent traversal after normalization
    - empty / absolute / drive-relative paths are rejected
    """
    if path is None:
        raise PathValidationError("path is required")
    if not isinstance(path, str):
        raise PathValidationError(f"path must be a string, got {type(path).__name__}")
    raw = path.strip()
    if not raw:
        raise PathValidationError("path must be non-empty")
    if "\\" in raw:
        raise PathValidationError(f"path must use POSIX separators: {path!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        raise PathValidationError(f"path must be relative: {path!r}")
    parts = []
    for part in candidate.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise PathValidationError(f"path must not contain parent traversal: {path!r}")
        parts.append(part)
    if not parts:
        raise PathValidationError(f"path must identify a file beneath the project root: {path!r}")
    return "/".join(parts)


def _is_probably_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _mode_bits(mode: Optional[int]) -> Optional[int]:
    if mode is None:
        return None
    return stat.S_IMODE(mode)


def _format_mode(mode: Optional[int]) -> str:
    if mode is None:
        return "(none)"
    return oct(stat.S_IMODE(mode))


@dataclass(frozen=True)
class SubmittedFile:
    """Immutable before/after artifact captured from a terminal emit(files=[...])."""

    path: str
    operation: Operation
    before: Optional[bytes]
    after: Optional[bytes]
    before_mode: Optional[int]
    after_mode: Optional[int]
    before_symlink_target: Optional[str] = None
    after_symlink_target: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_submitted_path(self.path))
        if self.operation not in ("create", "modify", "delete"):
            raise ValueError(f"unsupported operation: {self.operation!r}")
        if self.operation == "create":
            if self.before is not None or self.before_symlink_target is not None:
                raise ValueError("create artifacts must not include before state")
            if self.after is None and self.after_symlink_target is None:
                raise ValueError("create artifacts require after content or symlink target")
        elif self.operation == "delete":
            if self.after is not None or self.after_symlink_target is not None:
                raise ValueError("delete artifacts must not include after state")
            if self.before is None and self.before_symlink_target is None:
                raise ValueError("delete artifacts require before content or symlink target")
        elif self.operation == "modify":
            if (
                self.before is None
                and self.before_symlink_target is None
                and self.after is None
                and self.after_symlink_target is None
                and self.before_mode == self.after_mode
            ):
                raise ValueError("modify artifacts require a content, symlink, or mode change")

    def open(self) -> BinaryIO:
        """Return a file-like object over immutable after bytes."""
        if self.operation == "delete":
            raise OverlaySubagentError(f"cannot open deleted artifact: {self.path}")
        if self.after_symlink_target is not None:
            raise OverlaySubagentError(f"cannot open symlink artifact as bytes: {self.path}")
        if self.after is None:
            raise OverlaySubagentError(f"artifact has no after bytes: {self.path}")
        return io.BytesIO(self.after)

    def text(self, encoding: str = "utf-8") -> str:
        if self.operation == "delete":
            raise OverlaySubagentError(f"cannot read deleted artifact text: {self.path}")
        if self.after_symlink_target is not None:
            raise OverlaySubagentError(f"cannot read symlink artifact as text: {self.path}")
        if self.after is None:
            raise OverlaySubagentError(f"artifact has no after bytes: {self.path}")
        return self.after.decode(encoding)

    def diff(self) -> str:
        """Return a deterministic unified / metadata diff for this artifact."""
        lines: list[str] = []
        path = self.path

        # Symlink metadata
        if self.before_symlink_target is not None or self.after_symlink_target is not None:
            if self.operation == "create":
                lines.append(f"symlink create {path}")
                lines.append(f"+++ link:{self.after_symlink_target}")
            elif self.operation == "delete":
                lines.append(f"symlink delete {path}")
                lines.append(f"--- link:{self.before_symlink_target}")
            else:
                lines.append(f"symlink modify {path}")
                lines.append(f"--- link:{self.before_symlink_target}")
                lines.append(f"+++ link:{self.after_symlink_target}")
            before_mode = _mode_bits(self.before_mode)
            after_mode = _mode_bits(self.after_mode)
            if before_mode != after_mode:
                lines.append(f"mode change {_format_mode(before_mode)} -> {_format_mode(after_mode)}")
            return "\n".join(lines) + "\n"

        before = self.before if self.before is not None else b""
        after = self.after if self.after is not None else b""
        before_mode = _mode_bits(self.before_mode)
        after_mode = _mode_bits(self.after_mode)

        if self.operation == "create":
            old_name = "/dev/null"
            new_name = path
        elif self.operation == "delete":
            old_name = path
            new_name = "/dev/null"
        else:
            old_name = path
            new_name = path

        # Binary summary when either side is non-text.
        if (self.before is not None and not _is_probably_text(self.before)) or (
            self.after is not None and not _is_probably_text(self.after)
        ):
            lines.append(f"Binary files {old_name} and {new_name} differ")
            lines.append(f"before_size={len(before)} after_size={len(after)}")
            if before_mode != after_mode:
                lines.append(f"mode change {_format_mode(before_mode)} -> {_format_mode(after_mode)}")
            return "\n".join(lines) + "\n"

        before_text = before.decode("utf-8").splitlines(keepends=True)
        after_text = after.decode("utf-8").splitlines(keepends=True)
        # Ensure trailing newline representation is stable for empty files.
        if before and not before.endswith(b"\n"):
            if before_text:
                before_text[-1] += "\n"
        if after and not after.endswith(b"\n"):
            if after_text:
                after_text[-1] += "\n"

        hunk = list(
            difflib.unified_diff(
                before_text,
                after_text,
                fromfile=old_name,
                tofile=new_name,
                lineterm="\n",
            )
        )
        if not hunk and before_mode == after_mode and self.operation == "modify":
            # Mode-only change without content change.
            lines.append(f"--- {old_name}")
            lines.append(f"+++ {new_name}")
            lines.append(f"mode change {_format_mode(before_mode)} -> {_format_mode(after_mode)}")
            return "\n".join(lines) + "\n"

        if not hunk and self.operation in ("create", "delete") and not before and not after:
            lines.append(f"--- {old_name}")
            lines.append(f"+++ {new_name}")
        else:
            lines.extend(line.rstrip("\n") for line in hunk)

        if before_mode != after_mode:
            lines.append(f"mode change {_format_mode(before_mode)} -> {_format_mode(after_mode)}")
        return ("\n".join(lines) + "\n") if lines else ""

    def _destination(self, root: Path) -> Path:
        root = root.resolve()
        # Resolve intermediate parents for symlink-escape checks, but do not
        # follow a final symlink component. Apply must operate on the submitted
        # path itself when it is already a symlink.
        relative = Path(self.path)
        parent = (root / relative.parent).resolve() if relative.parent != Path(".") else root
        try:
            parent.relative_to(root)
        except ValueError as exc:
            raise PathValidationError(f"destination escapes root: {self.path}") from exc
        dest = parent / relative.name
        # Reject when an intermediate symlink caused parent to leave root.
        if root not in dest.parents and dest != root / relative.name and not str(dest).startswith(str(root) + '/'):
            raise PathValidationError(f"destination escapes root: {self.path}")
        return dest

    def _current_state(self, dest: Path) -> tuple[bool, Optional[bytes], Optional[int], Optional[str]]:
        # Path.exists() follows symlinks; detect broken/present symlinks via is_symlink().
        if not dest.is_symlink() and not dest.exists():
            return False, None, None, None
        mode = dest.lstat().st_mode
        if dest.is_symlink():
            return True, None, mode, str(dest.readlink())
        if dest.is_dir():
            raise ApplyConflict(self.path, "destination is a directory")
        return True, dest.read_bytes(), mode, None

    def _matches_before(self, dest: Path) -> Optional[str]:
        exists, data, mode, link = self._current_state(dest)
        before_exists = self.before is not None or self.before_symlink_target is not None or self.operation != "create"
        if self.operation == "create":
            if exists:
                return "destination already exists"
            return None

        if not exists:
            return "destination no longer matches submitted before state"

        if self.before_symlink_target is not None:
            if link is None:
                return "destination type or symlink target changed"
            if link != self.before_symlink_target:
                return "destination type or symlink target changed"
        else:
            if link is not None:
                return "destination type or symlink target changed"
            if data != self.before:
                return "destination no longer matches submitted before state"

        if _mode_bits(mode) != _mode_bits(self.before_mode):
            # Only enforce mode when before_mode was captured.
            if self.before_mode is not None:
                return "destination mode no longer matches submitted before state"
        return None

    def apply(self, root: Optional[Path] = None) -> None:
        """Apply this artifact under root after verifying before-state."""
        root_path = Path(root) if root is not None else Path.cwd()
        dest = self._destination(root_path)
        conflict = self._matches_before(dest)
        if conflict is not None:
            raise ApplyConflict(self.path, conflict)

        if self.operation == "delete":
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.after_symlink_target is not None:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(self.after_symlink_target)
        else:
            import tempfile

            assert self.after is not None
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{dest.name}.overlay-",
                dir=dest.parent,
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(self.after)
                if self.after_mode is not None:
                    mode = _mode_bits(self.after_mode)
                    assert mode is not None
                    tmp.chmod(mode)
                os.replace(tmp, dest)
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
            return

        if self.after_mode is not None and not dest.is_symlink():
            dest.chmod(_mode_bits(self.after_mode) or 0o644)


def classify_operation(
    *,
    before_exists: bool,
    after_exists: bool,
) -> Operation:
    if not before_exists and after_exists:
        return "create"
    if before_exists and not after_exists:
        return "delete"
    if before_exists and after_exists:
        return "modify"
    raise PathValidationError("submitted path missing from both before and after sides")


def response_diff(files: Mapping[str, SubmittedFile], paths: Optional[Iterable[str]] = None) -> str:
    """Generate a stable, path-sorted combined diff for submitted artifacts."""
    selected: list[SubmittedFile]
    if paths is None:
        selected = [files[k] for k in sorted(files)]
    else:
        wanted = [normalize_submitted_path(p) for p in paths]
        missing = [p for p in wanted if p not in files]
        if missing:
            raise KeyError(f"unknown submitted paths: {missing}")
        selected = [files[p] for p in wanted]

    chunks = [artifact.diff() for artifact in selected]
    # Keep a blank line between file diffs when both are non-empty.
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        if out and not out[-1].endswith("\n\n"):
            out.append("")
        out.append(chunk.rstrip("\n"))
    if not out:
        return ""
    return "\n".join(out) + "\n"


def apply_submitted_files(
    files: Mapping[str, SubmittedFile],
    *,
    paths: Optional[Iterable[str]] = None,
    root: Optional[Path] = None,
) -> None:
    """Conflict-check all selected artifacts, then apply them.

    Multi-file application is not transactionally atomic, but ordinary
    conflicts are collected during preflight so they do not cause partial
    writes.
    """
    if paths is None:
        selected_paths = sorted(files)
    else:
        selected_paths = [normalize_submitted_path(p) for p in paths]
        missing = [p for p in selected_paths if p not in files]
        if missing:
            raise KeyError(f"unknown submitted paths: {missing}")

    root_path = Path(root) if root is not None else Path.cwd()
    artifacts = [files[p] for p in selected_paths]

    conflicts: list[ApplyConflict] = []
    for artifact in artifacts:
        dest = artifact._destination(root_path)
        reason = artifact._matches_before(dest)
        if reason is not None:
            conflicts.append(ApplyConflict(artifact.path, reason))
    if conflicts:
        raise ApplyConflict(conflicts[0].path, conflicts[0].reason, conflicts=conflicts)

    for artifact in artifacts:
        artifact.apply(root=root_path)


@dataclass
class OverlaySubagentResponseBase:
    """Immutable submitted-artifact response helpers."""

    result: str
    files: Mapping[str, SubmittedFile]
    progress: list[str] | None = None
    turns: int = 0
    done: bool = True
    is_error: bool = False
    error: Optional[str] = None
    submission_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.progress is None:
            self.progress = []
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def diff(self, paths: Optional[Iterable[str]] = None) -> str:
        return response_diff(self.files, paths=paths)

    def apply(
        self,
        paths: Optional[Iterable[str]] = None,
        root: Optional[Path] = None,
    ) -> None:
        apply_submitted_files(self.files, paths=paths, root=root)



class SubmissionError(OverlaySubagentError):
    """Raised when terminal file submission cannot be materialized."""


@dataclass(frozen=True)
class PathSideState:
    """Captured existence/content/mode/symlink state for one path side."""

    exists: bool
    data: Optional[bytes] = None
    mode: Optional[int] = None
    symlink_target: Optional[str] = None
    is_dir: bool = False


def _lexists(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def capture_path_state(path: Path) -> PathSideState:
    """Capture metadata for a path without following a final symlink."""
    if not _lexists(path):
        return PathSideState(exists=False)
    mode = path.lstat().st_mode
    if path.is_symlink():
        return PathSideState(
            exists=True,
            mode=mode,
            symlink_target=str(path.readlink()),
        )
    if path.is_dir():
        return PathSideState(exists=True, mode=mode, is_dir=True)
    return PathSideState(exists=True, data=path.read_bytes(), mode=mode)


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a submitted relative path beneath project_root without following the final symlink."""
    normalized = normalize_submitted_path(relative_path)
    root = project_root.resolve()
    relative = Path(normalized)
    parent = (root / relative.parent).resolve() if relative.parent != Path(".") else root
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise PathValidationError(f"path escapes project root: {relative_path!r}") from exc
    dest = parent / relative.name
    if not (str(dest) == str(root / relative.name) or str(dest).startswith(str(root) + '/')):
        raise PathValidationError(f"path escapes project root: {relative_path!r}")
    return dest


def normalize_submission_paths(paths: Iterable[str]) -> list[str]:
    """Normalize and deterministically deduplicate submitted paths."""
    if paths is None or isinstance(paths, (str, bytes)):
        raise PathValidationError("files must be an iterable of path strings, not a string")
    seen: set[str] = set()
    ordered: list[str] = []
    for item in paths:
        normalized = normalize_submitted_path(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def materialize_submitted_files(
    paths: Iterable[str],
    *,
    project_root: Path,
    lower_root: Optional[Path] = None,
    per_file_limit: int = DEFAULT_PER_FILE_LIMIT,
    aggregate_limit: int = DEFAULT_AGGREGATE_LIMIT,
) -> dict[str, SubmittedFile]:
    """Materialize immutable before/after artifacts for explicitly submitted paths.

    `project_root` is the merged worker view. `lower_root` is the pre-worker /
    lower-side view used for `before` capture. When omitted, `before` is treated
    as missing (useful for pure create submissions in tests).
    """
    normalized = normalize_submission_paths(paths)
    project_root = Path(project_root)
    lower = Path(lower_root) if lower_root is not None else None

    artifacts: dict[str, SubmittedFile] = {}
    aggregate = 0

    for rel in normalized:
        merged_path = resolve_project_path(project_root, rel)
        after_state = capture_path_state(merged_path)

        if after_state.is_dir:
            raise SubmissionError(f"directories are not supported in initial submission: {rel}")

        if lower is None:
            before_state = PathSideState(exists=False)
        else:
            before_path = resolve_project_path(lower, rel)
            before_state = capture_path_state(before_path)
            if before_state.is_dir:
                raise SubmissionError(f"directories are not supported in initial submission: {rel}")

        operation = classify_operation(
            before_exists=before_state.exists,
            after_exists=after_state.exists,
        )

        before_size = len(before_state.data or b"")
        after_size = len(after_state.data or b"")
        if before_size > per_file_limit or after_size > per_file_limit:
            raise SubmissionError(
                f"submitted file exceeds per-file limit ({per_file_limit} bytes): {rel}"
            )
        aggregate += before_size + after_size
        if aggregate > aggregate_limit:
            raise SubmissionError(
                f"submitted files exceed aggregate limit ({aggregate_limit} bytes)"
            )

        artifacts[rel] = SubmittedFile(
            path=rel,
            operation=operation,
            before=before_state.data if before_state.symlink_target is None else None,
            after=after_state.data if after_state.symlink_target is None else None,
            before_mode=before_state.mode,
            after_mode=after_state.mode,
            before_symlink_target=before_state.symlink_target,
            after_symlink_target=after_state.symlink_target,
        )

    return artifacts


def submitted_files_to_payload(files: Mapping[str, SubmittedFile]) -> list[dict]:
    """Serialize submitted artifacts for the subagent socket protocol."""
    payload = []
    for path in sorted(files):
        artifact = files[path]
        payload.append(
            {
                "path": artifact.path,
                "operation": artifact.operation,
                "before": artifact.before,
                "after": artifact.after,
                "before_mode": artifact.before_mode,
                "after_mode": artifact.after_mode,
                "before_symlink_target": artifact.before_symlink_target,
                "after_symlink_target": artifact.after_symlink_target,
            }
        )
    return payload


def submitted_files_from_payload(payload: Optional[Iterable[dict]]) -> dict[str, SubmittedFile]:
    """Deserialize submitted artifacts received over the subagent socket."""
    if not payload:
        return {}
    if isinstance(payload, (str, bytes, Mapping)):
        raise SubmissionError("artifact payload must be an iterable of mappings")
    files: dict[str, SubmittedFile] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise SubmissionError(
                f"artifact payload item {index} must be a mapping"
            )
        try:
            artifact = SubmittedFile(
                path=item["path"],
                operation=item["operation"],
                before=item.get("before"),
                after=item.get("after"),
                before_mode=item.get("before_mode"),
                after_mode=item.get("after_mode"),
                before_symlink_target=item.get("before_symlink_target"),
                after_symlink_target=item.get("after_symlink_target"),
            )
        except (KeyError, TypeError, ValueError, PathValidationError) as exc:
            raise SubmissionError(
                f"invalid artifact payload item {index}: {exc}"
            ) from exc
        if artifact.path in files:
            raise SubmissionError(
                f"duplicate artifact payload path: {artifact.path}"
            )
        files[artifact.path] = artifact
    return files




# ---------------------------------------------------------------------------
# Ctypes / Linux kernel and namespace integration
# ---------------------------------------------------------------------------

class OverlayRuntimeError(OverlaySubagentError):
    """Raised when Linux namespace or mount operations fail."""

    def __init__(
        self,
        op: str,
        message: str,
        *,
        errno_val: Optional[int] = None,
        paths: Optional[dict[str, str]] = None,
        mount_options: Optional[str] = None,
        runtime_id: Optional[str] = None,
        mountinfo: Optional[str] = None,
    ):
        self.op = op
        self.errno_val = errno_val
        self.paths = paths or {}
        self.mount_options = mount_options
        self.runtime_id = runtime_id
        self.mountinfo = mountinfo

        parts = [f"Overlay runtime operation {op!r} failed: {message}"]
        if errno_val is not None:
            parts.append(f"errno: {errno_val}")
        if runtime_id:
            parts.append(f"runtime_id: {runtime_id}")
        if paths:
            paths_str = ", ".join(f"{k}={v}" for k, v in paths.items())
            parts.append(f"paths: {paths_str}")
        if mount_options:
            parts.append(f"mount_options: {mount_options}")
        if mountinfo:
            parts.append("/proc/self/mountinfo excerpt:")
            parts.append(str(mountinfo))

        super().__init__("\n".join(parts))


CLONE_NEWUSER = 0x10000000
CLONE_NEWNS = 0x00020000
MS_REC = 16384
MS_PRIVATE = 262144
MS_BIND = 4096
MS_RDONLY = 1
MS_REMOUNT = 32
MS_NOSUID = 2
MS_NODEV = 4
MNT_DETACH = 2
PR_SET_NO_NEW_PRIVS = 38
PR_SET_PDEATHSIG = 1


def _get_libc():
    import ctypes
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc
    except Exception as exc:
        raise OverlayRuntimeError("load_libc", f"failed to load libc.so.6: {exc}") from exc


def _read_mountinfo_excerpt() -> Optional[str]:
    try:
        path = Path("/proc/self/mountinfo")
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-10:])
    except Exception:
        pass
    return None


def overlay_capability_diagnostics() -> dict[str, Any]:
    """Return non-destructive host capability diagnostics for overlay workers."""
    import platform
    import shutil

    diagnostics: dict[str, Any] = {
        "platform": platform.system(),
        "procfs": Path("/proc/self/mountinfo").is_file(),
        "user_namespace_api": Path("/proc/self/uid_map").is_file(),
        "copy_command": shutil.which("cp"),
        "unprivileged_userns_clone": None,
    }
    setting = Path("/proc/sys/kernel/unprivileged_userns_clone")
    if setting.is_file():
        try:
            diagnostics["unprivileged_userns_clone"] = (
                setting.read_text(encoding="utf-8").strip() != "0"
            )
        except OSError:
            diagnostics["unprivileged_userns_clone"] = None
    return diagnostics


def require_overlay_capabilities() -> None:
    """Fail early for hosts that cannot possibly run this Linux overlay backend."""
    diagnostics = overlay_capability_diagnostics()
    missing = []
    if diagnostics["platform"] != "Linux":
        missing.append("Linux")
    if not diagnostics["procfs"]:
        missing.append("mounted procfs")
    if not diagnostics["user_namespace_api"]:
        missing.append("user namespace procfs API")
    if not diagnostics["copy_command"]:
        missing.append("cp command")
    if diagnostics["unprivileged_userns_clone"] is False:
        missing.append("kernel.unprivileged_userns_clone=1")
    if missing:
        detail = ", ".join(missing)
        raise OverlayRuntimeError(
            "capability_probe",
            f"required overlay capabilities unavailable: {detail}; "
            f"diagnostics={diagnostics!r}",
        )


def unshare_user_and_mount_namespaces(runtime_id: str = "") -> None:
    import ctypes
    libc = _get_libc()
    res = libc.unshare(CLONE_NEWUSER | CLONE_NEWNS)
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "unshare",
            "unshare(CLONE_NEWUSER | CLONE_NEWNS) failed",
            errno_val=err,
            runtime_id=runtime_id,
        )


def configure_id_maps(uid: int, gid: int, runtime_id: str = "") -> None:
    try:
        Path("/proc/self/uid_map").write_text(f"0 {uid} 1\n", encoding="utf-8")
        try:
            Path("/proc/self/setgroups").write_text("deny\n", encoding="utf-8")
        except Exception:
            pass
        Path("/proc/self/gid_map").write_text(f"0 {gid} 1\n", encoding="utf-8")
    except Exception as exc:
        raise OverlayRuntimeError(
            "configure_id_maps",
            f"failed to write UID/GID mapping: {exc}",
            paths={"uid_map": "/proc/self/uid_map", "gid_map": "/proc/self/gid_map"},
            runtime_id=runtime_id,
        ) from exc


def unshare_mount_namespace(runtime_id: str = "") -> None:
    import ctypes
    libc = _get_libc()
    res = libc.unshare(CLONE_NEWNS)
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "unshare_mount_namespace",
            "unshare(CLONE_NEWNS) failed",
            errno_val=err,
            runtime_id=runtime_id,
        )


def make_mounts_private(runtime_id: str = "") -> None:
    import ctypes
    libc = _get_libc()
    res = libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None)
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "make_mounts_private",
            "mount('/', MS_REC | MS_PRIVATE) failed",
            errno_val=err,
            runtime_id=runtime_id,
            mountinfo=_read_mountinfo_excerpt(),
        )


def bind_mount(source: Path, target: Path, runtime_id: str = "") -> None:
    import ctypes
    libc = _get_libc()
    res = libc.mount(
        str(source).encode(),
        str(target).encode(),
        None,
        MS_BIND,
        None,
    )
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "bind_mount",
            os.strerror(err),
            errno_val=err,
            paths={"source": str(source), "target": str(target)},
            runtime_id=runtime_id,
            mountinfo=_read_mountinfo_excerpt(),
        )


def _lowerdir_option(lower_dirs: Iterable[Path]) -> str:
    layers = [str(Path(path)) for path in lower_dirs]
    if not layers:
        raise ValueError("overlay requires at least one lower directory")
    if any(":" in layer or "," in layer for layer in layers):
        raise ValueError("overlay layer paths must not contain ':' or ','")
    return ":".join(layers)


def overlay_layer_usage(source_dir: Path) -> tuple[int, int]:
    """Return apparent bytes and inode count without following symlinks."""
    total_bytes = 0
    total_inodes = 0
    stack = [Path(source_dir)]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                total_inodes += 1
                info = entry.stat(follow_symlinks=False)
                total_bytes += info.st_size
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
    return total_bytes, total_inodes


def validate_snapshot_capacity(
    source_dir: Path,
    target_parent: Path,
    *,
    byte_limit: Optional[int],
    inode_limit: Optional[int],
    min_free_bytes: int,
    runtime_id: str = "",
) -> tuple[int, int]:
    bytes_used, inodes_used = overlay_layer_usage(source_dir)
    if byte_limit is not None and bytes_used > byte_limit:
        raise OverlayRuntimeError(
            "snapshot_preflight",
            f"upper layer exceeds snapshot byte limit ({bytes_used} > {byte_limit})",
            paths={"source": str(source_dir)},
            runtime_id=runtime_id,
        )
    if inode_limit is not None and inodes_used > inode_limit:
        raise OverlayRuntimeError(
            "snapshot_preflight",
            f"upper layer exceeds snapshot inode limit ({inodes_used} > {inode_limit})",
            paths={"source": str(source_dir)},
            runtime_id=runtime_id,
        )
    filesystem = os.statvfs(target_parent)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    required = bytes_used + min_free_bytes
    if free_bytes < required:
        raise OverlayRuntimeError(
            "snapshot_preflight",
            f"insufficient free space for snapshot ({free_bytes} available, {required} required)",
            paths={"source": str(source_dir), "target": str(target_parent)},
            runtime_id=runtime_id,
        )
    return bytes_used, inodes_used


def clone_overlay_layer(
    source_dir: Path,
    target_dir: Path,
    runtime_id: str = "",
    timeout: Optional[float] = DEFAULT_SNAPSHOT_TIMEOUT,
) -> None:
    import subprocess

    try:
        result = subprocess.run(
            [
                "cp",
                "-a",
                "--reflink=auto",
                f"{source_dir}/.",
                str(target_dir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise OverlayRuntimeError(
            "clone_overlay_layer",
            f"snapshot copy exceeded timeout ({timeout} seconds)",
            paths={"source": str(source_dir), "target": str(target_dir)},
            runtime_id=runtime_id,
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "copy command failed"
        raise OverlayRuntimeError(
            "clone_overlay_layer",
            detail,
            errno_val=None,
            paths={"source": str(source_dir), "target": str(target_dir)},
            runtime_id=runtime_id,
        )


def mount_overlay(
    lower_dirs: Iterable[Path],
    upper_dir: Path,
    work_dir: Path,
    target_dir: Path,
    runtime_id: str = "",
) -> None:
    import ctypes
    libc = _get_libc()
    lower_option = _lowerdir_option(lower_dirs)
    options = f"lowerdir={lower_option},upperdir={upper_dir},workdir={work_dir}"
    res = libc.mount(
        b"overlay",
        str(target_dir).encode(),
        b"overlay",
        MS_NOSUID | MS_NODEV,
        options.encode(),
    )
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "mount_overlay",
            "mount overlayfs failed",
            errno_val=err,
            paths={
                "lower": lower_option,
                "upper": str(upper_dir),
                "work": str(work_dir),
                "target": str(target_dir),
            },
            mount_options=options,
            runtime_id=runtime_id,
            mountinfo=_read_mountinfo_excerpt(),
        )


def mount_lower_view(
    lower_dirs: Iterable[Path],
    target_dir: Path,
    runtime_id: str = "",
) -> None:
    lower_dirs = [Path(path) for path in lower_dirs]
    if len(lower_dirs) == 1:
        bind_mount(lower_dirs[0], target_dir, runtime_id)
        import ctypes
        libc = _get_libc()
        res = libc.mount(
            None,
            str(target_dir).encode(),
            None,
            MS_BIND | MS_REMOUNT | MS_RDONLY,
            None,
        )
        if res != 0:
            err = ctypes.get_errno()
            raise OverlayRuntimeError(
                "mount_lower_view",
                "remount read-only lower view failed",
                errno_val=err,
                paths={"lower": str(lower_dirs[0]), "target": str(target_dir)},
                runtime_id=runtime_id,
                mountinfo=_read_mountinfo_excerpt(),
            )
        return

    import ctypes
    libc = _get_libc()
    lower_option = _lowerdir_option(lower_dirs)
    options = f"lowerdir={lower_option}"
    res = libc.mount(
        b"overlay",
        str(target_dir).encode(),
        b"overlay",
        MS_NOSUID | MS_NODEV,
        options.encode(),
    )
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "mount_lower_view",
            "mount read-only overlay lower view failed",
            errno_val=err,
            paths={"lower": lower_option, "target": str(target_dir)},
            mount_options=options,
            runtime_id=runtime_id,
            mountinfo=_read_mountinfo_excerpt(),
        )


def unmount_overlay(target_dir: Path, runtime_id: str = "") -> None:
    import ctypes
    import errno
    import time

    target = str(target_dir)
    libc = _get_libc()
    res = libc.umount2(target.encode(), 0)
    if res != 0:
        err = ctypes.get_errno()
        if err != errno.EBUSY:
            raise OverlayRuntimeError(
                "unmount",
                os.strerror(err),
                errno_val=err,
                paths={"target": target},
                runtime_id=runtime_id,
                mountinfo=_read_mountinfo_excerpt(),
            )
        res = libc.umount2(target.encode(), MNT_DETACH)
        if res != 0:
            err = ctypes.get_errno()
            raise OverlayRuntimeError(
                "unmount",
                os.strerror(err),
                errno_val=err,
                paths={"target": target},
                runtime_id=runtime_id,
                mountinfo=_read_mountinfo_excerpt(),
            )

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            mounted = any(
                len(fields := line.split()) > 4 and fields[4] == target
                for line in Path("/proc/self/mountinfo").read_text(
                    encoding="utf-8"
                ).splitlines()
            )
        except OSError:
            mounted = False
        if not mounted:
            return
        time.sleep(0.01)

    raise OverlayRuntimeError(
        "unmount",
        "mount remained present after unmount",
        errno_val=errno.EBUSY,
        paths={"target": target},
        runtime_id=runtime_id,
        mountinfo=_read_mountinfo_excerpt(),
    )


def set_no_new_privs(runtime_id: str = "") -> None:
    import ctypes
    libc = _get_libc()
    res = libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "set_no_new_privs",
            "prctl(PR_SET_NO_NEW_PRIVS) failed",
            errno_val=err,
            runtime_id=runtime_id,
        )


def set_parent_death_signal(expected_parent_pid: int, runtime_id: str = "") -> None:
    """Request SIGTERM if the process that spawned this worker exits."""
    import ctypes
    import signal

    libc = _get_libc()
    res = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise OverlayRuntimeError(
            "set_parent_death_signal",
            "prctl(PR_SET_PDEATHSIG, SIGTERM) failed",
            errno_val=err,
            runtime_id=runtime_id,
        )
    if os.getppid() != expected_parent_pid:
        raise OverlayRuntimeError(
            "set_parent_death_signal",
            "owner exited before parent-death signaling was configured",
            runtime_id=runtime_id,
        )


def start_parent_liveness_monitor(
    liveness_fd: int,
    expected_parent_pid: int,
    cleanup,
    *,
    runtime_id: str = "",
    exit_func=None,
    configure_death_signal: bool = True,
):
    """Tie this process to its direct owner using PDEATHSIG and pipe EOF."""
    import signal
    import threading

    if exit_func is None:
        exit_func = os._exit

    if configure_death_signal:
        set_parent_death_signal(expected_parent_pid, runtime_id)
    shutdown_lock = threading.Lock()
    shutting_down = False

    def owner_gone(*_args) -> None:
        nonlocal shutting_down
        with shutdown_lock:
            if shutting_down:
                return
            shutting_down = True
        try:
            cleanup()
        finally:
            try:
                os.close(liveness_fd)
            except OSError:
                pass
            exit_func(1)

    signal.signal(signal.SIGTERM, owner_gone)

    def monitor() -> None:
        try:
            while os.read(liveness_fd, 1):
                pass
        except OSError:
            pass
        owner_gone()

    thread = threading.Thread(
        target=monitor,
        name=f"overlay-owner-{runtime_id or 'worker'}",
        daemon=True,
    )
    thread.start()
    return thread


def setup_overlay_worker(config: Mapping[str, Any]) -> None:
    runtime_id = str(config["runtime_id"])
    home = Path(config["home"])
    lower_home = Path(config["lower_home"])
    lower_view = Path(config["lower_view"])
    upper_dir = Path(config["upper_dir"])
    work_dir = Path(config["work_dir"])
    project_cwd = Path(config["project_cwd"])
    lower_sources = [Path(path) for path in config["lower_sources"]]

    if config.get("unshare", True):
        host_uid = os.getuid()
        host_gid = os.getgid()
        unshare_user_and_mount_namespaces(runtime_id)
        configure_id_maps(host_uid, host_gid, runtime_id)
    else:
        unshare_mount_namespace(runtime_id)
    make_mounts_private(runtime_id)

    if not config.get("unshare", True):
        # A nested worker inherits the parent's mounted /home when it creates
        # its private mount namespace. Remove that view before mounting the
        # flattened lower chain; otherwise overlayfs sees an active stacked
        # mount and can alias lowerdirs with an upper/work directory.
        os.chdir("/")
        unmount_overlay(home, runtime_id)
        unmount_overlay(Path(config["inherited_lower_view"]), runtime_id)

    if config.get("bind_initial_lower", False):
        bind_mount(Path(config["initial_lower_source"]), lower_home, runtime_id)
        lower_sources = [lower_home]
    mount_lower_view(lower_sources, lower_view, runtime_id)
    mount_overlay(lower_sources, upper_dir, work_dir, home, runtime_id)
    set_no_new_privs(runtime_id)
    os.chdir(project_cwd)


def direct_child_pids(pid: Optional[int] = None) -> list[int]:
    """Return live direct child PIDs from procfs."""
    owner = os.getpid() if pid is None else pid
    children_path = Path(f"/proc/{owner}/task/{owner}/children")
    try:
        values = children_path.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return []
    return [int(value) for value in values]


def descendant_pids(pid: int) -> list[int]:
    """Return all live descendants of pid."""
    descendants: list[int] = []
    stack = direct_child_pids(pid)
    while stack:
        child = stack.pop()
        descendants.append(child)
        stack.extend(direct_child_pids(child))
    return descendants


def assert_overlay_quiescent(
    managed_child_pids: Iterable[int] = (),
    infrastructure_pids: Iterable[int] = (),
    runtime_id: str = "",
) -> None:
    """Reject sealing while processes outside fixed worker infrastructure run."""
    managed = set(managed_child_pids)
    infrastructure = set(infrastructure_pids)
    direct = set(direct_child_pids())
    unmanaged = sorted(direct - managed - infrastructure)
    for pid in infrastructure:
        unmanaged.extend(descendant_pids(pid))
    unmanaged = sorted(set(unmanaged))
    if unmanaged:
        raise OverlayRuntimeError(
            "seal_quiescence",
            "unmanaged child processes are still running: "
            + ", ".join(str(pid) for pid in unmanaged),
            runtime_id=runtime_id,
        )




class _WorkerOverlayOwner:
    def __init__(
        self,
        runtime_config,
        model,
        max_turns,
        spawn_worker,
        stop_worker,
        infrastructure_pids,
    ):
        self.project_root = Path(runtime_config["project_cwd"])
        self.model = model
        self.max_turns = max_turns
        self.snapshot_byte_limit = runtime_config["snapshot_byte_limit"]
        self.snapshot_inode_limit = runtime_config["snapshot_inode_limit"]
        self.snapshot_timeout = runtime_config["snapshot_timeout"]
        self.snapshot_min_free_bytes = runtime_config["snapshot_min_free_bytes"]
        self.lower_view_dir = Path(runtime_config["lower_view"])
        self.lower_sources = [
            Path(path) for path in runtime_config["lower_sources"]
        ]
        self.upper_dir = Path(runtime_config["upper_dir"])
        self.work_dir = Path(runtime_config["work_dir"])
        self.runtime_dir = self.upper_dir.parent
        self.runtime_config = runtime_config
        self._spawn_worker = spawn_worker
        self._stop_worker = stop_worker
        self._infrastructure_pids = infrastructure_pids
        self._children = []
        self._seal_generation = 0
        self._closed = False

    def _create_child(
        self,
        cwd,
        model,
        max_turns,
        snapshot_byte_limit,
        snapshot_inode_limit,
        snapshot_timeout,
        snapshot_min_free_bytes,
        recursive,
    ):
        return OverlaySubagent(
            cwd=cwd or str(self.project_root),
            model=model or self.model,
            max_turns=max_turns,
            _parent_overlay=self,
            snapshot_byte_limit=snapshot_byte_limit,
            snapshot_inode_limit=snapshot_inode_limit,
            snapshot_timeout=snapshot_timeout,
            snapshot_min_free_bytes=snapshot_min_free_bytes,
            recursive=recursive,
        )

    def _seal(self):
        import shutil

        managed_child_pids = [
            child._transport._proc.pid
            for child in self._children
            if child._transport._proc is not None
            and child._transport._proc.poll() is None
        ]
        assert_overlay_quiescent(
            managed_child_pids=managed_child_pids,
            infrastructure_pids=self._infrastructure_pids,
            runtime_id=str(self.runtime_config["runtime_id"]),
        )
        generation = self._seal_generation + 1
        sealed_upper = self.runtime_dir / f"sealed-child-{generation}"
        sealed_upper.mkdir()
        try:
            validate_snapshot_capacity(
                self.upper_dir,
                self.runtime_dir,
                byte_limit=self.snapshot_byte_limit,
                inode_limit=self.snapshot_inode_limit,
                min_free_bytes=self.snapshot_min_free_bytes,
                runtime_id=str(self.runtime_config["runtime_id"]),
            )
            clone_overlay_layer(
                self.upper_dir,
                sealed_upper,
                str(self.runtime_config["runtime_id"]),
                timeout=self.snapshot_timeout,
            )
        except Exception:
            shutil.rmtree(sealed_upper, ignore_errors=True)
            raise

        self._seal_generation = generation
        return [sealed_upper, *self.lower_sources]

    def _spawn_child_worker(self, child, bootstrap):
        pid = self._spawn_worker(
            bootstrap,
            str(child.project_root),
            child.runtime_config["session_db"],
        )
        child._transport._proc = _InheritedProcess(pid)

    def _release_child_worker(self, pid):
        if not self._closed:
            self._stop_worker(pid)

    def close(self):
        if self._closed:
            return
        for child in reversed(list(self._children)):
            child.close()
        self._children.clear()
        self._closed = True


def _overlay_repl_code(recursive: bool) -> str:
    code = r"""
def _overlay_request(action, **kwargs):
    global _request_id
    _request_id += 1
    req_id = _request_id
    import json as _json
    _send_tool_request(_json.dumps({
        "tool": "__overlay_child__",
        "args": {"action": action, **kwargs},
        "request_id": req_id,
    }))
    return _wait_for_ack(req_id)


class OverlayChildResponse:
    def __init__(self, handle):
        self._handle = handle

    def _status(self):
        return _overlay_request("response_status", response=self._handle)

    @property
    def result(self):
        return self._status()["result"]

    @property
    def files(self):
        return self._status()["files"]

    @property
    def progress(self):
        return self._status()["progress"]

    @property
    def turns(self):
        return self._status()["turns"]

    @property
    def done(self):
        return self._status()["done"]

    @property
    def is_error(self):
        return self._status()["is_error"]

    @property
    def error(self):
        return self._status()["error"]

    @property
    def submission_error(self):
        return self._status()["submission_error"]

    def wait(self, timeout=None):
        _overlay_request("response_wait", response=self._handle, timeout=timeout)
        return self

    def diff(self, paths=None):
        return _overlay_request("response_diff", response=self._handle, paths=paths)

    def apply(self, paths=None):
        return _overlay_request("response_apply", response=self._handle, paths=paths)


class OverlaySubagent:
    def __init__(
        self,
        cwd=None,
        model=None,
        max_turns=50,
        snapshot_byte_limit=1073741824,
        snapshot_inode_limit=250000,
        snapshot_timeout=120.0,
        snapshot_min_free_bytes=536870912,
        recursive=False,
    ):
        if not __OVERLAY_RECURSIVE__:
            raise RuntimeError(
                "recursive overlay subagents require OverlaySubagent(recursive=True)"
            )
        self._handle = _overlay_request(
            "child_create",
            cwd=cwd,
            model=model,
            max_turns=max_turns,
            snapshot_byte_limit=snapshot_byte_limit,
            snapshot_inode_limit=snapshot_inode_limit,
            snapshot_timeout=snapshot_timeout,
            snapshot_min_free_bytes=snapshot_min_free_bytes,
            recursive=recursive,
        )
        self._closed = False

    def send(self, prompt, *, bg=False, max_turns=None, timeout=None):
        response = _overlay_request(
            "child_send",
            child=self._handle,
            prompt=prompt,
            bg=bg,
            max_turns=max_turns,
            timeout=timeout,
        )
        return OverlayChildResponse(response)

    def close(self):
        if self._closed:
            return
        _overlay_request("child_close", child=self._handle)
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


import code_agent.overlay_subagent as _overlay_subagent_module
_overlay_subagent_module.OverlaySubagent = OverlaySubagent


def emit(value, release=False, files=None):
    if files is not None and not release:
        raise ValueError("files may only be provided with release=True")
    global _request_id
    _request_id += 1
    req_id = _request_id
    msg_type = "emit" if release else "progress"
    _send_output(msg_type, str(value) + "\n")
    import json as _json
    _send_tool_request(_json.dumps({
        "tool": "__emit__",
        "args": {"value": value, "release": release, "files": files},
        "request_id": req_id
    }))
    _wait_for_ack(req_id)
"""
    return code.replace("__OVERLAY_RECURSIVE__", repr(recursive), 1)


def _build_overlay_worker_code() -> str:
    from code_agent.subagent import WORKER_CODE

    code = WORKER_CODE.replace(
        "def worker_main(port, authkey, model, max_turns):",
        "def worker_main(port, authkey, model, max_turns, runtime_config):",
        1,
    )
    code = code.replace(
        "    # Connect to host",
        """    import os
    import subprocess
    from code_agent.overlay_subagent import (
        set_parent_death_signal,
        setup_overlay_worker,
        start_parent_liveness_monitor,
    )

    overlay_children = {}
    overlay_owner = None

    def _stop_overlay_child(pid):
        entry = overlay_children.pop(pid, None)
        if entry is None:
            return False
        child, liveness_write_fd = entry
        try:
            os.close(liveness_write_fd)
        except OSError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return True

    def _shutdown_overlay_children():
        if overlay_owner is not None:
            overlay_owner.close()
        for pid in reversed(list(overlay_children)):
            _stop_overlay_child(pid)

    def _spawn_overlay_child(bootstrap, cwd, session_db):
        liveness_read_fd, liveness_write_fd = os.pipe()
        env = os.environ.copy()
        env["OVERLAY_LIVENESS_FD"] = str(liveness_read_fd)
        env["OVERLAY_OWNER_PID"] = str(os.getpid())
        env["CODE_AGENT_SESSION_DB"] = session_db
        try:
            child = subprocess.Popen(
                [sys.executable, "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                start_new_session=True,
                pass_fds=(liveness_read_fd,),
                env=env,
            )
        except Exception:
            os.close(liveness_read_fd)
            os.close(liveness_write_fd)
            raise
        os.close(liveness_read_fd)
        child.stdin.write(bootstrap.encode())
        child.stdin.close()
        overlay_children[child.pid] = (child, liveness_write_fd)
        return child.pid

    _overlay_liveness_fd = int(os.environ["OVERLAY_LIVENESS_FD"])
    _overlay_owner_pid = int(os.environ["OVERLAY_OWNER_PID"])
    _overlay_runtime_id = str(runtime_config["runtime_id"])

    # Creating a user namespace fails once another thread exists. Configure
    # PDEATHSIG and complete namespace setup before starting the monitor.
    set_parent_death_signal(_overlay_owner_pid, _overlay_runtime_id)
    setup_overlay_worker(runtime_config)
    start_parent_liveness_monitor(
        _overlay_liveness_fd,
        _overlay_owner_pid,
        _shutdown_overlay_children,
        runtime_id=_overlay_runtime_id,
        configure_death_signal=False,
    )

    # ToolREPL uses multiprocessing fork. Fork clears PDEATHSIG and removes the
    # monitor thread, so re-arm both protections in every ToolREPL child.
    import code_agent.repl_agent as _overlay_repl_agent
    _ordinary_tool_worker_main = _overlay_repl_agent._tool_worker_main

    def _overlay_tool_worker_main(*args, **kwargs):
        tool_owner_pid = os.getppid()
        set_parent_death_signal(tool_owner_pid, _overlay_runtime_id)
        start_parent_liveness_monitor(
            _overlay_liveness_fd,
            tool_owner_pid,
            lambda: None,
            runtime_id=_overlay_runtime_id,
            configure_death_signal=False,
        )
        return _ordinary_tool_worker_main(*args, **kwargs)

    _overlay_repl_agent._tool_worker_main = _overlay_tool_worker_main

    # Connect to host""",
        1,
    )
    code = code.replace(
        "    from code_agent.agent import CodeAgent",
        """    from code_agent.agent import CodeAgent
    from code_agent.overlay_subagent import (
        _WorkerOverlayOwner,
        _overlay_repl_code,
        assert_overlay_quiescent,
        direct_child_pids,
        materialize_submitted_files,
        submitted_files_to_payload,
    )""",
        1,
    )
    code = code.replace(
        """            super().__init__()
            self.output_hook = self._subagent_output_hook""",
        """            self._overlay_runtime_config = runtime_config
            self._overlay_emit_injected = False
            self._overlay_submitted_files = None
            self._overlay_owner = None
            self._overlay_children = {}
            self._overlay_responses = {}
            self._overlay_next_handle = 0
            super().__init__()
            self.output_hook = self._overlay_output_hook""",
        1,
    )
    code = code.replace(
        """    # Create agent
    agent = SubagentWorker(sock, model, max_turns)
""",
        """    # Create agent
    agent = SubagentWorker(sock, model, max_turns)
    if runtime_config["recursive"]:
        ok, message = agent.attach_skill("overlay_subagent_worker")
        if not ok:
            raise RuntimeError(message)
""",
        1,
    )
    worker_adapter_start = "        # Disable CLI display hooks, but report turns to the parent."
    worker_adapter_end = "    # Create agent"
    adapter_start = code.index(worker_adapter_start)
    adapter_end = code.index(worker_adapter_end, adapter_start)
    new_worker_adapter = """        def _get_tool_repl(self):
            repl = super()._get_tool_repl()
            if not self._overlay_emit_injected:
                repl._inject_code(
                    _overlay_repl_code(self._overlay_runtime_config["recursive"])
                )
                self._overlay_emit_injected = True
            worker = getattr(repl, "_worker", None)
            if worker is not None and worker.pid is not None:
                overlay_infrastructure_pids.add(worker.pid)
            return repl

        @staticmethod
        def _deserialize_overlay_value(value):
            if isinstance(value, dict) and "__b64__" in value:
                import base64
                return base64.b64decode(value["__b64__"])
            if isinstance(value, list):
                return [SubagentWorker._deserialize_overlay_value(item) for item in value]
            if isinstance(value, dict):
                return {
                    key: SubagentWorker._deserialize_overlay_value(item)
                    for key, item in value.items()
                }
            return value

        def _overlay_output_hook(self, value, release):
            if not release:
                _send_msg(
                    self._host_sock,
                    ("progress", str(value) if value is not None else ""),
                )
                return

            files = self._overlay_submitted_files
            self._overlay_submitted_files = None
            payload = None
            submission_error = None
            if files is not None:
                try:
                    artifacts = materialize_submitted_files(
                        files,
                        project_root=self._overlay_runtime_config["project_cwd"],
                        lower_root=self._overlay_runtime_config["lower_project_root"],
                    )
                    payload = submitted_files_to_payload(artifacts)
                except Exception as exc:
                    submission_error = f"{type(exc).__name__}: {exc}"
            _send_msg(
                self._host_sock,
                ("result", {
                    "value": str(value) if value is not None else "",
                    "files": payload,
                    "submission_error": submission_error,
                }),
            )

        def on_repl_execute(self, code):
            super().on_repl_execute(code)
            self._turn_count += 1
            _send_msg(self._host_sock, ("turn", self._turn_count))

        def _new_overlay_handle(self, prefix):
            self._overlay_next_handle += 1
            return f"{prefix}-{self._overlay_next_handle}"

        def _overlay_response_status(self, response):
            return {
                "result": response.result if response.done else "",
                "files": sorted(response.files) if response.done else [],
                "progress": response.progress,
                "turns": response.turns,
                "done": response.done,
                "is_error": response.is_error if response.done else False,
                "error": response.error if response.done else None,
                "submission_error": (
                    response.submission_error if response.done else None
                ),
            }

        def _handle_overlay_child_request(self, args):
            action = args.get("action")
            if action == "child_create":
                child = self._overlay_owner._create_child(
                    cwd=args.get("cwd"),
                    model=args.get("model"),
                    max_turns=args["max_turns"],
                    snapshot_byte_limit=args["snapshot_byte_limit"],
                    snapshot_inode_limit=args["snapshot_inode_limit"],
                    snapshot_timeout=args["snapshot_timeout"],
                    snapshot_min_free_bytes=args["snapshot_min_free_bytes"],
                    recursive=args["recursive"],
                )
                handle = self._new_overlay_handle("child")
                self._overlay_children[handle] = child
                return handle
            if action == "child_send":
                child = self._overlay_children[args["child"]]
                response = child.send(
                    args.get("prompt", ""),
                    bg=bool(args.get("bg", False)),
                    max_turns=args.get("max_turns"),
                    timeout=args.get("timeout"),
                )
                handle = self._new_overlay_handle("response")
                self._overlay_responses[handle] = response
                return handle
            if action == "child_close":
                handle = args["child"]
                child = self._overlay_children.pop(handle)
                child.close()
                return None
            response = self._overlay_responses[args["response"]]
            if action == "response_status":
                return self._overlay_response_status(response)
            if action == "response_wait":
                response.wait(args.get("timeout"))
                return self._overlay_response_status(response)
            if action == "response_diff":
                return response.diff(paths=args.get("paths"))
            if action == "response_apply":
                response.apply(paths=args.get("paths"))
                return None
            raise ValueError(f"unknown overlay child action: {action!r}")

        def _handle_tool_request(self, repl, req):
            if req.get("tool") == "__overlay_child__":
                request_id = req.get("request_id")
                args = self._deserialize_overlay_value(req.get("args", {}))
                try:
                    result = self._handle_overlay_child_request(args)
                    repl.send_reply(request_id, result=result)
                except Exception as exc:
                    repl.send_reply(
                        request_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                finally:
                    if request_id is not None:
                        repl.send_ack(request_id)
                return
            if req.get("tool") == "__emit__":
                args = req.get("args", {})
                release = bool(args.get("release", False))
                self._overlay_submitted_files = (
                    self._deserialize_overlay_value(args.get("files"))
                    if release
                    else None
                )
            return super()._handle_tool_request(repl, req)

"""
    code = code[:adapter_start] + new_worker_adapter + code[adapter_end:]
    task_start_marker = '            if cmd_type == "task":'
    task_end_marker = '            elif cmd_type == "shutdown":'
    task_start = code.index(task_start_marker)
    task_end = code.index(task_end_marker, task_start)
    new_task = """            if cmd_type == "task":
                prompt = cmd_data.get("prompt", "")
                task_max_turns = cmd_data.get("max_turns", max_turns)
                agent._turn_count = 0
                agent._overlay_submitted_files = None

                try:
                    agent.run_interaction(prompt, max_turns=task_max_turns)
                except KeyboardInterrupt:
                    _send_msg(sock, ("error", "Task interrupted"))
                except Exception as e:
                    import traceback
                    _send_msg(sock, ("error", f"{type(e).__name__}: {e}\\n{traceback.format_exc()}"))
                finally:
                    agent.complete = False
                    agent._final_result = None

"""
    code = code[:task_start] + new_task + code[task_end:]
    code = code.replace(
        """    # Main loop - receive tasks
""",
        """    overlay_infrastructure_pids = set(direct_child_pids())
    overlay_owner = _WorkerOverlayOwner(
        runtime_config,
        model,
        max_turns,
        lambda bootstrap, cwd, session_db: _spawn_overlay_child(
            bootstrap,
            cwd,
            session_db,
        ),
        _stop_overlay_child,
        overlay_infrastructure_pids,
    )
    agent._overlay_owner = overlay_owner
    # Main loop - receive tasks
""",
        1,
    )
    code = code.replace(
        """            elif cmd_type == "shutdown":
                break
""",
        """            elif cmd_type == "shutdown":
                _shutdown_overlay_children()
                break
""",
        1,
    )
    return code


OVERLAY_WORKER_CODE = _build_overlay_worker_code()


class OverlaySubagentResponse:
    def __init__(self, agent: "OverlaySubagent"):
        self._agent_ref = weakref.ref(agent)
        self._result = ""
        self._files: Mapping[str, SubmittedFile] = MappingProxyType({})
        self._progress: list[str] = []
        self._turns = 0
        self._done = False
        self._is_error = False
        self._error: Optional[str] = None
        self._submission_error: Optional[str] = None

    @property
    def result(self) -> str:
        self._refresh()
        return self._result

    @property
    def files(self) -> Mapping[str, SubmittedFile]:
        self._refresh()
        return self._files

    @property
    def progress(self) -> list[str]:
        self._refresh()
        return list(self._progress)

    @property
    def turns(self) -> int:
        self._refresh()
        return self._turns

    @property
    def done(self) -> bool:
        self._refresh()
        return self._done

    @property
    def is_error(self) -> bool:
        self._refresh()
        return self._is_error

    @property
    def error(self) -> Optional[str]:
        self._refresh()
        return self._error

    @property
    def submission_error(self) -> Optional[str]:
        self._refresh()
        return self._submission_error

    def _agent(self) -> "OverlaySubagent":
        agent = self._agent_ref()
        if agent is None:
            raise RuntimeError("OverlaySubagent has been closed")
        return agent

    def _refresh(self) -> None:
        if not self._done:
            self._agent()._poll()

    def diff(self, paths: Optional[Iterable[str]] = None) -> str:
        return response_diff(self.files, paths=paths)

    def apply(
        self,
        paths: Optional[Iterable[str]] = None,
        root: Optional[Path] = None,
    ) -> None:
        destination = root if root is not None else self._agent().project_root
        apply_submitted_files(self.files, paths=paths, root=destination)

    def wait(self, timeout: Optional[float] = None) -> "OverlaySubagentResponse":
        import time
        start = time.time()
        while not self.done:
            if timeout is not None and time.time() - start > timeout:
                break
            time.sleep(0.1)
        return self

    def __repr__(self) -> str:
        status = "running"
        if self.done:
            status = "error" if self.is_error else "complete"
        return (
            f"<OverlaySubagentResponse status='{status}', turns={self.turns}, "
            f"progress_updates={len(self.progress)}, files={len(self.files)}>"
        )


class _InheritedProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.stderr = None
        self.stdout = None

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 1
        try:
            stat_text = Path(f"/proc/{self.pid}/stat").read_text(
                encoding="utf-8"
            )
            state = stat_text.rsplit(")", 1)[1].lstrip()[0]
            if state == "Z":
                return 1
        except (OSError, IndexError):
            return 1
        return None

    def kill(self):
        if self.poll() is not None:
            return
        try:
            os.killpg(self.pid, 9)
        except ProcessLookupError:
            pass

    def wait(self, timeout=None):
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
        return None


class _BoundedPipeCapture:
    """Continuously drain a pipe while retaining only its trailing bytes."""

    def __init__(self, stream, limit: int = 64 * 1024):
        import threading

        self._stream = stream
        self._limit = limit
        self._buffer = bytearray()
        self._truncated = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain,
            name="overlay-worker-output",
            daemon=True,
        )
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.read(8192)
                if not chunk:
                    return
                with self._lock:
                    self._buffer.extend(chunk)
                    overflow = len(self._buffer) - self._limit
                    if overflow > 0:
                        del self._buffer[:overflow]
                        self._truncated = True
        except (OSError, ValueError):
            return
        finally:
            try:
                self._stream.close()
            except OSError:
                pass

    def text(self) -> str:
        with self._lock:
            data = bytes(self._buffer)
            truncated = self._truncated
        text = data.decode("utf-8", errors="replace").strip()
        if truncated:
            text = "[truncated to trailing output]\n" + text
        return text

    def join(self, timeout: float = 1.0) -> None:
        self._thread.join(timeout)


class OverlaySubagent:
    """Persistent subagent with an isolated, sealable home overlay."""

    def __init__(
        self,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 50,
        _parent_overlay: Optional["OverlaySubagent"] = None,
        snapshot_byte_limit: Optional[int] = DEFAULT_SNAPSHOT_BYTE_LIMIT,
        snapshot_inode_limit: Optional[int] = DEFAULT_SNAPSHOT_INODE_LIMIT,
        snapshot_timeout: Optional[float] = DEFAULT_SNAPSHOT_TIMEOUT,
        snapshot_min_free_bytes: int = DEFAULT_SNAPSHOT_MIN_FREE_BYTES,
        recursive: bool = False,
    ):
        import tempfile
        import uuid
        from code_agent.subagent import Subagent

        require_overlay_capabilities()
        self.id = str(uuid.uuid4())[:8]
        self.project_root = Path(cwd or Path.cwd()).resolve()
        self.model = model or Subagent.default_model
        self.max_turns = max_turns
        self._parent_overlay = _parent_overlay
        self.snapshot_byte_limit = snapshot_byte_limit
        self.snapshot_inode_limit = snapshot_inode_limit
        self.snapshot_timeout = snapshot_timeout
        self.snapshot_min_free_bytes = snapshot_min_free_bytes
        self.recursive = recursive
        self._children: list[OverlaySubagent] = []
        self._current_response: Optional[OverlaySubagentResponse] = None
        self._seal_generation = 0
        self._closed = False
        self._liveness_write_fd: Optional[int] = None
        self._stdout_capture: Optional[_BoundedPipeCapture] = None
        self._stderr_capture: Optional[_BoundedPipeCapture] = None

        self.home = Path.home().resolve()
        self.merged_dir = self.home
        try:
            self.project_relative = self.project_root.relative_to(self.home)
        except ValueError as exc:
            raise PathValidationError(
                f"overlay project must be beneath the user's home directory: {self.project_root}"
            ) from exc

        self.runtime_dir = Path(tempfile.mkdtemp(prefix=f"overlay_subagent_{self.id}_"))
        self.upper_dir = self.runtime_dir / "upper"
        self.work_dir = self.runtime_dir / "work"
        self.lower_home_dir = self.runtime_dir / "lower-home"
        self.lower_view_dir = self.runtime_dir / "lower-view"
        for path in (
            self.upper_dir,
            self.work_dir,
            self.lower_home_dir,
            self.lower_view_dir,
        ):
            path.mkdir()

        if _parent_overlay is None:
            self.lower_sources = [self.lower_home_dir]
        else:
            try:
                self.lower_sources = _parent_overlay._seal()
            except Exception:
                import shutil
                shutil.rmtree(self.runtime_dir, ignore_errors=True)
                raise
        self.lower_dir = self.lower_view_dir / self.project_relative
        self.runtime_config = {
            "runtime_id": self.id,
            "home": str(self.home),
            "lower_sources": [str(path) for path in self.lower_sources],
            "bind_initial_lower": _parent_overlay is None,
            "initial_lower_source": str(self.home),
            "session_db": str(self.runtime_dir / "sessions.db"),
            "lower_home": str(self.lower_home_dir),
            "lower_view": str(self.lower_view_dir),
            "inherited_lower_view": (
                str(_parent_overlay.lower_view_dir)
                if _parent_overlay is not None
                else None
            ),
            "upper_dir": str(self.upper_dir),
            "work_dir": str(self.work_dir),
            "project_cwd": str(self.project_root),
            "lower_project_root": str(self.lower_dir),
            "unshare": _parent_overlay is None,
            "snapshot_byte_limit": self.snapshot_byte_limit,
            "snapshot_inode_limit": self.snapshot_inode_limit,
            "snapshot_timeout": self.snapshot_timeout,
            "snapshot_min_free_bytes": self.snapshot_min_free_bytes,
            "recursive": self.recursive,
        }
        self._transport = Subagent(
            cwd=str(self.project_root),
            model=model,
            max_turns=max_turns,
        )
        if _parent_overlay is not None:
            _parent_overlay._children.append(self)

    def _worker_bootstrap(self, port: int, authkey: bytes) -> str:
        import sys
        return f"""
import sys
sys.path = {repr(sys.path)}
exec({repr(OVERLAY_WORKER_CODE)})
worker_main(
    {port},
    bytes.fromhex({repr(authkey.hex())}),
    {repr(self._transport.model)},
    {self.max_turns},
    {repr(self.runtime_config)},
)
"""

    def _ensure_started(self) -> None:
        import fcntl
        import socket
        import subprocess

        transport = self._transport
        if transport._started and transport._proc and transport._proc.poll() is None:
            return
        if transport._proc:
            transport._cleanup()

        authkey = os.urandom(16)
        transport._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        transport._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        transport._server.bind(("127.0.0.1", 0))
        transport._server.listen(1)
        port = transport._server.getsockname()[1]
        transport._server.settimeout(30)
        bootstrap = self._worker_bootstrap(port, authkey)

        if self._parent_overlay is None:
            liveness_read_fd, liveness_write_fd = os.pipe()
            env = os.environ.copy()
            env["OVERLAY_LIVENESS_FD"] = str(liveness_read_fd)
            env["OVERLAY_OWNER_PID"] = str(os.getpid())
            env["CODE_AGENT_SESSION_DB"] = self.runtime_config["session_db"]
            try:
                transport._proc = subprocess.Popen(
                    [os.sys.executable, "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self.project_root),
                    start_new_session=True,
                    pass_fds=(liveness_read_fd,),
                    env=env,
                )
            except Exception:
                os.close(liveness_read_fd)
                os.close(liveness_write_fd)
                raise
            assert transport._proc.stdout is not None
            assert transport._proc.stderr is not None
            self._stdout_capture = _BoundedPipeCapture(transport._proc.stdout)
            self._stderr_capture = _BoundedPipeCapture(transport._proc.stderr)
            os.close(liveness_read_fd)
            self._liveness_write_fd = liveness_write_fd
            transport._proc.stdin.write(bootstrap.encode())
            transport._proc.stdin.close()
        else:
            self._parent_overlay._spawn_child_worker(self, bootstrap)

        try:
            transport._conn, _ = transport._server.accept()
            transport._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            from code_agent.subagent import _recv_msg, _send_msg
            if _recv_msg(transport._conn) != authkey:
                raise RuntimeError("Overlay subagent authentication failed")
        except Exception:
            if transport._conn is not None:
                try:
                    transport._conn.close()
                except Exception:
                    pass
                transport._conn = None
            proc = transport._proc
            if proc is not None:
                try:
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                        except AttributeError:
                            proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                            proc.wait()
                except Exception:
                    pass
                transport._proc = None
            transport._started = False
            if self._liveness_write_fd is not None:
                try:
                    os.close(self._liveness_write_fd)
                except OSError:
                    pass
                self._liveness_write_fd = None
            if self._stderr_capture is not None:
                self._stderr_capture.join()
            if self._stdout_capture is not None:
                self._stdout_capture.join()
            stderr = self._stderr_capture.text() if self._stderr_capture else ""
            stdout = self._stdout_capture.text() if self._stdout_capture else ""
            detail = "\n".join(
                part for part in (
                    f"stderr:\n{stderr}" if stderr else "",
                    f"stdout:\n{stdout}" if stdout else "",
                ) if part
            )
            if detail:
                raise RuntimeError(
                    f"Overlay subagent failed to start.\n{detail}"
                )
            raise
        finally:
            transport._server.close()
            transport._server = None

        from code_agent.subagent import _IncrementalMessageReceiver, _send_msg
        _send_msg(transport._conn, "ok")
        fd = transport._conn.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        transport._receiver = _IncrementalMessageReceiver()
        transport._started = True


    def send(
        self,
        prompt: str,
        *,
        bg: bool = False,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> OverlaySubagentResponse:
        from code_agent.subagent import _send_msg, _wrap_subagent_task

        self._ensure_started()
        if self._current_response is not None and not self._current_response.done:
            raise RuntimeError("OverlaySubagent already has a running task")

        response = OverlaySubagentResponse(self)
        self._current_response = response
        _send_msg(self._transport._conn, ("task", {
            "prompt": _wrap_subagent_task(prompt),
            "max_turns": max_turns or self.max_turns,
        }))
        if not bg:
            try:
                response.wait(timeout)
            except KeyboardInterrupt:
                print(
                    "\nOverlay subagent task is still running in the background. "
                    "Use subagent.last to inspect or wait for it."
                )
                raise
        return response

    def _poll(self) -> None:
        import socket
        from code_agent.subagent import _NO_MESSAGE

        response = self._current_response
        conn = self._transport._conn
        if response is None or response._done or conn is None:
            return

        while True:
            try:
                message = self._transport._receiver.receive(conn)
                if message is _NO_MESSAGE:
                    return
                msg_type, data = message
            except (socket.timeout, BlockingIOError):
                return
            except ConnectionError as exc:
                if self._stderr_capture is not None:
                    self._stderr_capture.join()
                if self._stdout_capture is not None:
                    self._stdout_capture.join()
                stderr = self._stderr_capture.text() if self._stderr_capture else ""
                stdout = self._stdout_capture.text() if self._stdout_capture else ""
                details = [str(exc)]
                if stderr:
                    details.append("stderr:\n" + stderr)
                if stdout:
                    details.append("stdout:\n" + stdout)
                response._error = "\n".join(details)
                response._is_error = True
                response._result = response._error
                response._done = True
                return

            if msg_type == "progress":
                response._progress.append(str(data) if data is not None else "")
            elif msg_type == "turn":
                response._turns = int(data)
            elif msg_type == "result":
                if isinstance(data, dict):
                    response._result = data.get("value", "")
                    response._files = MappingProxyType(
                        submitted_files_from_payload(data.get("files"))
                    )
                    response._submission_error = data.get("submission_error")
                else:
                    response._result = str(data) if data is not None else ""
                response._done = True
                return
            elif msg_type == "error":
                response._error = f"Subagent task failed:\n\n{data}"
                response._is_error = True
                response._result = response._error
                response._done = True
                return


    @property
    def last(self) -> Optional[OverlaySubagentResponse]:
        return self._current_response

    def close(self) -> None:
        if self._closed:
            return
        for child in reversed(list(self._children)):
            child.close()
        self._children.clear()
        process = self._transport._proc
        inherited_pid = (
            process.pid
            if self._parent_overlay is not None and process is not None
            else None
        )
        try:
            self._transport.close()
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, 9)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=2)
                        except Exception:
                            pass
                except Exception:
                    pass
            if inherited_pid is not None and not self._parent_overlay._closed:
                try:
                    self._parent_overlay._release_child_worker(inherited_pid)
                except Exception:
                    pass
            if self._liveness_write_fd is not None:
                try:
                    os.close(self._liveness_write_fd)
                except OSError:
                    pass
                self._liveness_write_fd = None
            if self._stderr_capture is not None:
                self._stderr_capture.join()
            if self._stdout_capture is not None:
                self._stdout_capture.join()
            import shutil
            for work_root in self.runtime_dir.glob("work*"):
                try:
                    work_root.chmod(0o700)
                except OSError:
                    pass
                nested_work = work_root / "work"
                try:
                    nested_work.chmod(0o700)
                except OSError:
                    pass
            shutil.rmtree(self.runtime_dir, ignore_errors=True)
            if self._parent_overlay is not None:
                try:
                    self._parent_overlay._children.remove(self)
                except ValueError:
                    pass
            self._closed = True

    def kill(self) -> str:
        for child in reversed(list(self._children)):
            child.kill()
        result = self._transport.kill()
        self.close()
        return result

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"[OverlaySubagent id={self.id} project={self.project_root}]"
