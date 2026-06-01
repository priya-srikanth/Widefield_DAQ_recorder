"""Launch wfield NeuroCAAS GUI with a PyQt progress-bar compatibility patch.

The released wfield 0.4.2 GUI can crash on newer PyQt when it passes a
numpy.float64 to QProgressBar.setValue during AWS uploads. This launcher keeps
the installed package unchanged and coerces progress values to plain ints.
"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QProgressBar


_original_set_value = QProgressBar.setValue


def _set_value_int(self, value):
    return _original_set_value(self, int(value))


def main() -> None:
    QProgressBar.setValue = _set_value_int

    from wfield.ncaas_gui import main as ncaas_main

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    ncaas_main(folder=folder)


if __name__ == "__main__":
    main()
