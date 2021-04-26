from src import features, models, ExperimentRunner
from src.features import preprocessing, sequences
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()



