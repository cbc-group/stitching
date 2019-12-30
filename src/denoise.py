import glob
import logging
import os

from csbdeep.utils import plot_history

import imageio
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V
import numpy as np

from utils import find_dataset_dir

logger = logging.getLogger(__name__)


def train(
    image,
    patch_shape=(16, 32, 32),
    ratio=0.7,
    name="untitled",
    model_dir=".",
    view_history=True,
):
    # create data generator object
    datagen = N2V_DataGenerator()

    image = image[np.newaxis, ..., np.newaxis]

    patches = datagen.generate_patches_from_list(image, shape=patch_shape)
    logger.info(f"patch shape: {patches.shape}")
    n_patches = patches.shape[0]
    logger.info(f"{n_patches} patches generated")

    # split training set and validation set
    i = int(n_patches * ratio)
    X, X_val = patches[:i], patches[i:]

    # create training config
    config = N2VConfig(
        X,
        unet_kern_size=3,
        train_steps_per_epoch=100,
        train_epochs=100,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=4,
        n2v_perc_pix=1.6,
        n2v_patch_shape=patch_shape,
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
    )

    model = N2V(config=config, name=name, basedir=model_dir)

    # train and save the model
    history = model.train(X, X_val)
    model.export_TF()

    plot_history(history, ["loss", "val_loss"])


def infer(images, name="untitled", model_dir="."):
    model = N2V(config=None, name=name, basedir=model_dir)

    for image in images:
        yield model.predict(image, axes="ZYX")


def run(file_list, train_on, name="untitled", dst_dir="."):
    try:
        os.makedirs(dst_dir)
    except FileExistsError:
        pass

    model_dir = os.path.join(dst_dir, "model")
    try:
        os.makedirs(model_dir)
    except FileExistsError:
        pass

    logger.info("preloading image for training")
    image = imageio.volread(train_on)
    train(image, name=name, model_dir=model_dir)

    def load_image():
        for file_path in file_list:
            yield imageio.volread(file_path)

    for file_name, result in zip(
        file_list, infer(load_image(), name=name, model_dir=model_dir)
    ):
        fname = os.path.basename(file_name)
        file_path = os.path.join(dst_dir, fname)
        imageio.imwrite(file_path, result)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    src_path = find_dataset_dir("4F/20191230/twin_light_sample")
    file_list = glob.glob(os.path.join(src_path, "*.tif"))

    dst_dir, _ = os.path.split(src_path)
    dst_dir = f"{dst_dir}_denoise"
    model_dir = f"{dst_dir}_model"

    run(file_list, train_on="2.tif")
