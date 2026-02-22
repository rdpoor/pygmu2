"""
examples_helper.py — shared utilities for pygmu2 example scripts.

Typical usage
-------------
    import pygmu2 as pg
    from examples_helper import s, pad_clip, run_demos

    pg.set_sample_rate(44100)

    DEMOS = [
        ("My demo", my_demo_fn),
    ]

    if __name__ == "__main__":
        run_demos(DEMOS)

Running a script
----------------
    python examples/my_eg.py          # interactive menu
    python examples/my_eg.py 1        # run demo 1 and quit
    python examples/my_eg.py a        # run all demos and quit

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import logging
import sys

import pygmu2 as pg

logger = logging.getLogger(__name__)


def s(secs: float) -> int:
    """Convert seconds to samples at the current global sample rate."""
    return pg.seconds_to_samples(secs, pg.get_sample_rate())


def pad_clip(pe, silence_secs: float = 0.5):
    """
    Return *pe* with a short silence appended so consecutive clips are audibly
    separated.  The caller is responsible for playing the returned PE.
    """
    n = pe.extent().end
    return pg.SetExtentPE(pe, 0, n + s(silence_secs))


def run_demos(demos: dict) -> None:
    """
    Run the standard interactive demo menu (or a single demo from argv).

    *demos* is an ``{name: callable}`` dict — identical to the ``DEMOS``
    variable used in every example script.

    Call as the body of ``if __name__ == "__main__":``::

        if __name__ == "__main__":
            run_demos(DEMOS)
    """

    import os
    # Enable DEBUG via PYGMU_EXAMPLES_DEBUG=1 environment variable
    if os.environ.get("PYGMU_EXAMPLES_DEBUG"):
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    def resolve_choice(choice: str):
        """Return (name, fn) for a valid digit choice, or (None, None)."""
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(demos):
                return demos[idx - 1]
        return None, None

    def print_menu():
        print("Available demos:")
        for i, (name, _fn) in enumerate(demos, start=1):
            print(f"  {i}: {name}")
        print("  ?: show list")
        print("  a: run all")
        print("  q: quit")

    def run_all():
        for _name, fn in demos:
            fn()

    def choose_and_play():
        while True:
            choice = input("Select demo number: ").strip()
            if choice.lower() == "q":
                break
            if choice.lower() == "a":
                run_all()
                continue
            if choice == "?":
                print_menu()
                continue
            _name, fn = resolve_choice(choice)
            if fn is not None:
                fn()
            else:
                print(f"Invalid choice '{choice}', '?' to see list")

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
        if choice == "a":
            run_all()
            raise SystemExit(0)
        _name, fn = resolve_choice(choice)
        if fn is not None:
            fn()
        else:
            print(f"Invalid choice '{choice}'")
    else:
        print_menu()
        choose_and_play()
