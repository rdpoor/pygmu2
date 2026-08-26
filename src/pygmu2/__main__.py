"""
Entry point for `python -m pygmu2`: print the package version.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pygmu2


def main():
    print(f"pygmu2 {pygmu2.__version__}")


if __name__ == "__main__":
    main()
