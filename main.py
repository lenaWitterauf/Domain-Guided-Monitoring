import logging
from src import _main

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib.font_manager").disabled = True


if __name__ == "__main__":
    _main()