import tensorflow as tf
import mlflow

class MLFlowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric('epoch', epoch)
        for log_key in logs.keys():
            mlflow.log_metric(log_key, logs[log_key])
