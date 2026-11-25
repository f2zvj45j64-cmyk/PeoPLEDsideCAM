#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local trigger client (Tkinter) to test JetsonDualCamPy Web Service.

Sends JSON (new format):
{
  "data1": {"position":"1",        "SN":"DM1_XXX",  "result":"NA", "failure symptoms":"NA"},
  "data2": {"position":"DM6_XXX",  "SN":"NA",      "result":"NA", "failure symptoms":"NA"},
  "data3": {"position":"DM11_XXX", "SN":"NA",      "result":"NA", "failure symptoms":"NA"}
}
"""

import json
import time
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import requests
except Exception as e:
    requests = None

def default_sn():
    return "SN" + time.strftime("%Y%m%d%H%M%S")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trigger Client (local)")
        self.geometry("820x420")

        if requests is None:
            ttk.Label(self, text="Missing 'requests'. Install: pip install requests").pack(pady=20)
            return

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        self.host = tk.StringVar(value="http://127.0.0.1:9000")
        self.pos1 = tk.StringVar(value="1")
        self.pos2 = tk.StringVar(value="DM6_XXX")
        self.pos3 = tk.StringVar(value="DM11_XXX")

        self.sn1 = tk.StringVar(value=default_sn())
        self.sn2 = tk.StringVar(value="NA")
        self.sn3 = tk.StringVar(value="NA")

        def row(r, label, var):
            ttk.Label(frm, text=label, width=18).grid(row=r, column=0, sticky="w", pady=6)
            ttk.Entry(frm, textvariable=var, width=60).grid(row=r, column=1, sticky="we", pady=6)

        row(0, "Server base URL", self.host)
        row(1, "data1 position", self.pos1)
        row(2, "data2 position", self.pos2)
        row(3, "data3 position", self.pos3)
        row(4, "data1 SN", self.sn1)
        row(5, "data2 SN", self.sn2)
        row(6, "data3 SN", self.sn3)

        btn = ttk.Button(frm, text="SEND /api/eval", command=self.send)
        btn.grid(row=7, column=0, columnspan=2, sticky="w", pady=12)

        self.out = tk.Text(frm, height=10)
        self.out.grid(row=8, column=0, columnspan=2, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(8, weight=1)

        ttk.Button(frm, text="New SN", command=self.new_sn).grid(row=7, column=1, sticky="e")

    def new_sn(self):
        self.sn1.set(default_sn())
        # keep others as NA by default
        self.sn2.set("NA")
        self.sn3.set("NA")

    def send(self):
        base = self.host.get().strip().rstrip("/")
        url = base + "/api/eval"
        payload = {
            "data1": {"position": self.pos1.get().strip(), "SN": self.sn1.get().strip(), "result": "NA", "failure symptoms": "NA"},
            "data2": {"position": self.pos2.get().strip(), "SN": self.sn2.get().strip(), "result": "NA", "failure symptoms": "NA"},
            "data3": {"position": self.pos3.get().strip(), "SN": self.sn3.get().strip(), "result": "NA", "failure symptoms": "NA"},
        }
        try:
            r = requests.post(url, json=payload, timeout=20)
            self.out.delete("1.0", tk.END)
            self.out.insert(tk.END, f"HTTP {r.status_code}\n")
            self.out.insert(tk.END, r.text + "\n")
            if r.status_code != 200:
                messagebox.showerror("Error", r.text)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()
