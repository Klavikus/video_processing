from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator


class NNDenoise:
    def __init__(self, config):
        self.config = config
        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory=config['TRAIN_IMAGES_DIRECTORY'])
        print(imgs[0].shape, imgs[1].shape)
        patch_shape = (config['N2V_PATCH_SHAPE'], config['N2V_PATCH_SHAPE'])
        self.X = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape)
        self.X_val = datagen.generate_patches_from_list(imgs[1:], shape=patch_shape)
        data_for_cfg = {
            'unet_kern_size': config['UNET_KERN_SIZE'],
            'train_steps_per_epoch': int(self.X.shape[0] / config['N2V_PATCH_SHAPE']),
            'train_epochs': config['TRAIN_EPOCHS'],
            'train_loss': config['TRAIN_LOSS'],
            'batch_norm': config['BATCH_NORM'],
            'train_batch_size': config['TRAIN_BATCH_SIZE'],
            'n2v_perc_pix': config['N2V_PERC_PIX'],
            'n2v_patch_shape': patch_shape,
            'n2v_manipulator': config['N2V_MANIPULATOR'],
            'n2v_neighborhood_radius': config['N2V_NEIGHBORHOOD_RADIUS'],
        }
        n2v_config = N2VConfig(
            self.X,
            **data_for_cfg,
        )

        model_name = config['MODEL_NAME']
        basedir = config['MODEL_DIRECTORY']
        self.model = N2V(n2v_config, model_name, basedir=basedir)

    # def generate_training_data(self, patch_shape=(128, 128)):
    #     datagen = N2V_DataGenerator()
    #     imgs = datagen.load_imgs_from_directory(directory=self._directory)
    #     print(imgs[0].shape, imgs[1].shape)
    #     X = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape)
    #     X_val = datagen.generate_patches_from_list(imgs[1:2], shape=patch_shape)

    def train(self):
        self.model.train(self.X, self.X_val)

    def predict(self, image):
        return self.model.predict(image, axes='YX')
