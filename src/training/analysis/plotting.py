import matplotlib.pyplot as plt
import tensorflow as tf
from ..models import BaseModel

class MetricPlotter:
    def __init__(self, model: BaseModel, plot_path: str = 'plots/'):
        self.model = model
        self.plot_path = plot_path
    
    def plot_all_metrics(self):
        self._plot_metric('loss')

        for metric in self.model.metrics:
            self._plot_metric(metric.name)

    def _plot_metric(self, metric_name: str):
        history = self.model.history.history

        plt.figure(figsize=(20, 10))
        plt.title(metric_name)
        plt.xlabel('epoch')
        plt.ylabel(metric_name)
        plt.plot(history[metric_name])
        if ('val_' + metric_name) in history:
            plt.plot(history['val_' + metric_name])
            plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.plot_path + metric_name + '.png')
