#!/usr/bin/env python3
"""Pack ``github/`` into ``workspace/dist/not_named_yet_github_upload.zip``.

Archive paths are relative to ``github/`` (unpack → ready-made repo root).

Uses ``git ls-files`` from the **workspace** root when ``.git`` exists (tracks paths under
``github/`` only). Otherwise scans disk under ``github/``.

Review ``docs/GITHUB_UPLOAD_ALLOWLIST.md`` before sharing."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def github_root_from_here() -> Path:
    """``github/scripts/<this>.py`` → ``github/``."""
    return Path(__file__).resolve().parents[1]


def workspace_root_from_here() -> Path:
    """Parent of ``github/``."""
    return Path(__file__).resolve().parents[2]


SKIP_DIR_NAMES = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    ".git",
    "dist",
})


def _skip_relative(rel: Path) -> bool:
    for part in rel.parts:
        if part in SKIP_DIR_NAMES:
            return True
        if part.endswith(".egg-info"):
            return True
    if rel.suffix.lower() in {".pyc", ".pyo", ".pyd"}:
        return True
    return False


def iter_github_files_fs(gh: Path) -> list[Path]:
    out: list[Path] = []
    for path in gh.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(gh)
        if _skip_relative(rel):
            continue
        out.append(path)
    out.sort(key=lambda x: x.as_posix())
    return out


def iter_github_files_git(ws: Path, gh: Path) -> list[Path] | None:
    if not (ws / ".git").is_dir():
        return None
    proc = subprocess.run(
        ["git", "-C", str(ws), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    gh_resolved = gh.resolve()
    paths: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode(errors="surrogateescape").replace("\\", "/")
        candidate = (ws / rel.replace("/", os.sep)).resolve()
        if not candidate.is_file():
            continue
        try:
            candidate.relative_to(gh_resolved)
        except ValueError:
            continue
        paths.append(candidate)
    paths.sort(key=lambda x: x.as_posix())
    return paths


def write_zip(github_root: Path, out_zip: Path, paths: list[Path], verbose: bool) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.is_file():
        out_zip.unlink()
    gh_resolved = github_root.resolve()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            arcname = path.resolve().relative_to(gh_resolved).as_posix()
            zf.write(path, arcname=arcname)
            if verbose:
                print(arcname)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zip github/ contents into a single archive (paths relative to github/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output zip path (default: dist/not_named_yet_github_upload.zip under workspace root)",
    )
    parser.add_argument(
        "--filesystem-only",
        action="store_true",
        help="Always scan github/ on disk; ignore git tracked list even if .git exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    gh = github_root_from_here()
    ws = workspace_root_from_here()

    source = "filesystem"
    paths: list[Path]
    if not args.filesystem_only:
        git_paths = iter_github_files_git(ws, gh)
        if git_paths is not None:
            paths = git_paths
            source = "git ls-files (github/)"
        else:
            paths = iter_github_files_fs(gh)
    else:
        paths = iter_github_files_fs(gh)

    out_zip = args.output or (ws / "dist" / "not_named_yet_github_upload.zip")
    write_zip(gh, out_zip, paths, args.verbose)
    mib = out_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_zip} ({len(paths)} files, {mib:.2f} MiB) [{source}]")


if __name__ == "__main__":
    main()
