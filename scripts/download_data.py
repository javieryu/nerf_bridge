import gdown
from pathlib import Path
from rich.prompt import Prompt, Confirm
from rich.console import Console
import argparse

CONSOLE_WIDTH = 100
console = Console(width=CONSOLE_WIDTH)

repo_dir = Path(__file__).parent.absolute().parent.absolute()
download_default_dir = repo_dir / "data/"

bags = {
    "desk": {
        "id": "1FT3lTKZxKd9z-7xzOmPu6JqmKj00rs49",
        "size": 1.85,
    },
    "test": {
        "id": "1urrEoZHUwz3rSbutV6msF69Dtbq9Hk3R",
        "size": 0.1,
    },
}

parser = argparse.ArgumentParser(description="Download ROSBags from Google Drive")
parser.add_argument(
    "--path",
    type=str,
    default=str(download_default_dir),
    help="Where to download the ROSBags",
)

def download_bag(bag_name, bag, download_dir):
    id = bag["id"]
    download_path = Path(download_dir) / bag_name
    console.rule(f"[bold cyan] Downloading {bag_name} ")
    gdown.download_folder(id=id, output=str(download_path), quiet=False)

if __name__ == "__main__":
    console.rule("[bold cyan] NerfBridge Sample ROSBag Downloader")
    console.print("[bold cyan] Available ROSBags: ")
    total_size = 0
    for key, value in bags.items():
        size = value["size"]
        total_size += size
        console.print(f"[red] {key} - {size} GB", justify="center")
    console.print(f"[bold red] all - {total_size} GB", justify="center")

    choices = list(bags.keys())
    choices.append("all")

    download_choice = Prompt.ask(
        "[bold cyan] Which rosbag do you want to download? ",
        console=console,
        choices=choices,
        show_choices=False,
    )
    
    download_path = parser.parse_args().path
    console.print(f"[bold][cyan] Preparing to download to: [/cyan] [red]{download_path}")
    confirmation = Confirm.ask("[bold cyan] Do you want to continue?")

    if confirmation:
        if download_choice == "all":
            for key, value in bags.items():
                download_bag(key, value, download_path)
        else:
            download_bag(download_choice, bags[download_choice], download_path)
