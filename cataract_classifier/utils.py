"""Contains utility functions to handle file information"""

from collections import defaultdict

from rich import print
from rich.table import Table


def sizeof_fmt(num, suffix="B"):
    """Converts a file size (in bytes) to a human-readable format (e.g., 1024 bytes → 1.0 KiB)."""
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def display_dir_items(path):
    """Displays the contents of a directory in a table, showing each item’s type (File/Folder) and size."""
    table = Table()
    table.add_column("Type", justify="right", no_wrap=True)
    table.add_column("Path", style="cyan")
    table.add_column("Size", justify="right", style="green")

    # Iterate through the directory and get information for each item
    filetype_counter = defaultdict(int)
    for filepath in path.glob("*"):
        pathtype = "File" if filepath.is_file() else "Folder"
        filetype_counter[pathtype] += 1
        filesize = (
            sizeof_fmt(filepath.stat().st_size) if pathtype == "File" else ""
        )
        table.add_row(pathtype, str(filepath), filesize)

    # Set the table title with the count of files and folders
    table.title = f"{filetype_counter['File']} files, {filetype_counter['Folder']} folders"

    # Print the table
    print(table)
