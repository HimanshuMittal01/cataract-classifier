"""
Contains utility functions to handle file information
"""

from collections import defaultdict

from rich import print
from rich.table import Table


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def display_dir_items(path):
    table = Table()
    table.add_column("Type", justify="right", no_wrap=True)
    table.add_column("Path", style="cyan")
    table.add_column("Size", justify="right", style="green")

    filetype_counter = defaultdict(int)
    for filepath in path.glob("*"):
        pathtype = "File" if filepath.is_file() else "Folder"
        filetype_counter[pathtype] += 1
        filesize = (
            sizeof_fmt(filepath.stat().st_size) if pathtype == "File" else ""
        )
        table.add_row(pathtype, str(filepath), filesize)

    table.title = f"{filetype_counter['File']} files, {filetype_counter['Folder']} folders"
    print(table)
