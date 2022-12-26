from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator


class NNDenoise:
    def __init__(self, directory):
        self._directory = directory
        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory=self._directory)
        print(imgs[0].shape, imgs[1].shape)
        factor = 4
        patch_shape = (32 * factor, 32 * factor)
        self.X = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape)
        self.X_val = datagen.generate_patches_from_list(imgs[1:], shape=patch_shape)
        data_for_cfg = {
            'unet_kern_size': 3,
            'train_steps_per_epoch': int(self.X.shape[0] / 128),
            'train_epochs': 50,
            'train_loss': 'mse',
            'batch_norm': True,
            'train_batch_size': 64,
            'n2v_perc_pix': 0.15,
            'n2v_patch_shape': (32 * factor, 32 * factor),
            'n2v_manipulator': 'uniform_withCP',
            'n2v_neighborhood_radius': 91,
        }
        config = N2VConfig(
            self.X,
            **data_for_cfg,
        )

        model_name = 'n2v_64x64_05_256_5_v2'
        basedir = 'models'
        self.model = N2V(config, model_name, basedir=basedir)

    def generate_training_data(self, patch_shape=(128, 128)):
        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory=self._directory)
        print(imgs[0].shape, imgs[1].shape)
        X = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape)
        X_val = datagen.generate_patches_from_list(imgs[1:2], shape=patch_shape)

    def train(self):
        self.model.train(self.X, self.X_val)

    def predict(self, image):
        return self.model.predict(image, axes='YX')
