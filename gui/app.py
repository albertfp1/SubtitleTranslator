import tkinter as tk
from tkinter import ttk


class App:
    """Minimal GUI wrapper for subtitle translation"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AISRT - AI Subtitle Translator")
        ttk.Label(self.root, text="GUI not yet implemented").pack(padx=20, pady=20)

    def mainloop(self):
        self.root.mainloop()
