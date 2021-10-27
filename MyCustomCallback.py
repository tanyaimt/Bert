import tensorflow as tf

class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager) -> None:
        self.ckpt_manager=ckpt_manager
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        print("Checkpoint guardado en {}.".format(self.checkpoint_path))
