import segmentation_models as sm
import tensorflow as tf


class UnetModel:
    def __init__(self):
        sm.set_framework('tf.keras')

        self.BACKBONE = 'efficientnetb3'
        self.ACTIVATION = 'sigmoid'
        self.BATCH_SIZE = 8
        self.NUM_CLASSES = 1
        self.LR = 1e-4
        self.EPOCHS = 50

    def compile(self, load_model_path):
        model = sm.Unet(self.BACKBONE, classes=self.NUM_CLASSES, activation=self.ACTIVATION)
        optim = tf.keras.optimizers.Adam(self.LR)
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True,
                                               mode='min'),
            tf.keras.callbacks.ReduceLROnPlateau()
        ]

        if load_model_path:
            model.load_weights(load_model_path)

        return model
