"""Folder watcher: which files may be run (spec 6.5; ticket 023).

Only regular CSV files whose real path sits inside the configured folder are accepted. The check
resolves symlinks and junctions first, so a link inside the folder that points outside is refused.
A path spelled with ".." is refused even when it resolves inside the folder.
"""

from pathlib import Path

CSV_SUFFIX = ".csv"


def accept(path: Path, folder: Path) -> bool:
    path = Path(path)
    if ".." in path.parts:
        return False
    real_folder = Path(folder).resolve()
    real_path = path.resolve()
    if not real_path.is_relative_to(real_folder):
        return False
    return real_path.is_file()


def scan(folder: Path) -> dict[Path, tuple[int, int]]:
    """Accepted CSV files directly in folder, mapped to (mtime_ns, size) so a replaced file shows up."""
    found: dict[Path, tuple[int, int]] = {}
    for path in Path(folder).iterdir():
        if path.suffix.lower() != CSV_SUFFIX or not accept(path, folder):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:  # removed between listing and stat
            continue
        found[path] = (stat.st_mtime_ns, stat.st_size)
    return found
