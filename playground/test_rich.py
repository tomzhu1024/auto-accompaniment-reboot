import random
import time

from rich.live import Live
from rich.table import Table
from rich.progress import Progress

def generate_table() -> Table:
    """Make a new table."""
    table = Table()
    table.add_column("ID")
    table.add_column("Value")
    table.add_column("Status")

    for row in range(random.randint(2, 6)):
        value = random.random() * 100
        table.add_row(
            f"{row}", f"{value:3.2f}", "[red]ERROR" if value < 50 else "[green]SUCCESS"
        )
    return table

print('before')

with Progress(transient=True) as progress:

    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:
        progress.update(task1, advance=5)
        progress.update(task2, advance=3)
        progress.update(task3, advance=9)
        time.sleep(0.02)
with Live(generate_table(), refresh_per_second=4, transient=True) as live:
    with Live(generate_table(), refresh_per_second=4, transient=True) as live2:
        for _ in range(10):
            time.sleep(0.4)
            live.update(generate_table())
            live2.update(generate_table())
print('after')
