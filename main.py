"""Traffic counting GUI entry point."""

from __future__ import annotations

import sys


def _show_dependency_error(error: Exception) -> None:
    message = (
        "缺少运行依赖，无法启动项目。\n\n"
        f"错误信息：{error}\n\n"
        "请在项目目录执行：\n"
        "python -m pip install -r requirements.txt\n\n"
        "然后重新运行 run.bat 或 python main.py。"
    )
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("依赖缺失", message)
        root.destroy()
    except Exception:
        print(message, file=sys.stderr)


def _check_dependencies() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        from PIL import Image, ImageTk  # noqa: F401
    except Exception as exc:
        _show_dependency_error(exc)
        return False
    return True


def main() -> int:
    if not _check_dependencies():
        return 1

    from traffic_counter.gui import TrafficCounterApp

    app = TrafficCounterApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
