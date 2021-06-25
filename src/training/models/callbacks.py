import tensorflow as tf
import numpy as np
import mlflow
import logging


class MLFlowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("epoch", epoch)
        for log_key in logs.keys():
            mlflow.log_metric(key=log_key, value=logs[log_key], step=epoch)


class BestModelRestoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric="val_loss", minimize=True):
        super(BestModelRestoreCallback, self).__init__()
        self.metric = metric
        self.minimize = minimize

    def on_train_begin(self, logs=None):
        self.best_weights = None
        self.best_metric_value = np.Inf if self.minimize else np.NINF
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        current_metric_value = logs.get(self.metric)
        if self._is_better(current_metric_value):
            logging.debug(
                "Model metric %s improved from %f to %f",
                self.metric,
                self.best_metric_value,
                current_metric_value,
            )
            self.best_metric_value = current_metric_value
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

    def _is_better(self, current_metric_value):
        if self.minimize:
            return np.less(current_metric_value, self.best_metric_value)
        else:
            return np.greater(current_metric_value, self.best_metric_value)

    def on_train_end(self, logs=None):
        if self.best_epoch > -1:
            logging.info(
                "Restoring best model weights with %s: %f from epoch %d",
                self.metric,
                self.best_metric_value,
                self.best_epoch,
            )
            self.model.set_weights(self.best_weights)
