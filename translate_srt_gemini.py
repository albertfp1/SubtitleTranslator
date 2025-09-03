import asyncio
import sys

from cli import cli_main
from gui import App

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if len(sys.argv) > 1:
        cli_main()
    else:
        app_instance = App()
        app_instance.mainloop()
