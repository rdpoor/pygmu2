"""
Entry point for running pygmu2 as a module

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import sys
from pygmu2.logger import set_global_logging, get_logger
from pygmu2 import hello, __version__


def main():
    """Main entry point"""
    # Set up logging
    logger = set_global_logging(level="INFO")
    logger.info(f"pygmu2 v{__version__} starting...")
    
    # Example usage
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "World"
    
    message = hello(name)
    print(message)
    logger.info("Application completed successfully")


if __name__ == "__main__":
    main()
