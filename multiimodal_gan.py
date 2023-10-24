import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Generator, Discriminator
from datasets import load_from_disk
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from models import weights_init_normal
from torchvision.transforms.functional import to_pil_image


class multi_gan(pl.LightningModule):
    def __init__(self,
                 img_size,
                 latent_dim,
                 channels,
                 ds_path,
                 batch_size,
                 shuffle,
                 num_workers,
                 lr_dis,
                 lr_gen,
                 *args,
                 **kwargs
                 ):
        super(multi_gan, self).__init__()

        self.save_hyperparameters()

        self.generator = Generator(img_size = self.hparams.img_size,

                                   latent_dim = self.hparams.latent_dim,

                                   channels = self.hparams.channels)

        self.discriminator = Discriminator(img_size = self.hparams.img_size,

                                           channels = self.hparams.channels)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # will be used to get images at end of epoch
        self.validation_z = self._get_noise(8, self.hparams.latent_dim)
        self.loss_adv = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False

    def setup(self, stage):

        def transform_apply(examples):
            examples["pixel_values"] = [transform_image(image.convert("RGB")) for image in examples["resized_image"]]
            return examples


        # load the dataset
        ds = load_from_disk(self.hparams.ds_path)
        if self.hparams.selective_test:
            ds = ds.shuffle().select(range(50))
        if self.hparams.shuffle:
            ds = ds.shuffle()
        # define the augmentations
        transform_image = transforms.Compose(
            [
                transforms.Resize((self.hparams.image_resize, self.hparams.image_resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        # set on the fly augmentations
        ds.set_transform(transform_apply)
        self.train_ds = ds


    def train_dataloader(self):
        def collate(examples):
            # from prev. step , each sample of example will have keys : pixel_values, resized_image
            # pixel_values will be tensor
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            return pixel_values

        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
            collate_fn=collate
        )

    def configure_optimizers(self):
        g_optim = Adam(self.generator.parameters(), lr=self.hparams.lr_gen, betas=(0.9, 0.999))
        d_optim = Adam(self.discriminator.parameters(), lr=self.hparams.lr_dis, betas=(0.9, 0.999))
        return [d_optim, g_optim], []

    def forward(self, z, *args, **kwargs):
        # given a latent vector z, generate an image
        return self.generator(z)

    def training_step(self, batch, batch_idx):

        # get optimizers
        opt_d, opt_g = self.optimizers()

        # get the real batch
        pixel_values_real = batch # [N, C, 512, 512]

        # Train discriminator
        result = None
        # if optimizer_idx == 0:
        # pass via the discriminator and calc. loss
        result = self._disc_step(pixel_values_real)
        _dict = {"loss_discriminator": result, "global_step":self.global_step}
        # optimize the discriminator
        opt_d.zero_grad()
        self.manual_backward(result)
        opt_d.step()


        # Train generator
        # if optimizer_idx == 1:
        # pass via the generator , then pass sample via discriminator and calc. loss
        result = self._gen_step(pixel_values_real)
        _dict.update({"loss_generator": result, "global_step":self.global_step})
        # optimize the generator
        opt_g.zero_grad()
        self.manual_backward(result)
        opt_g.step()

        self.log_dict(_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return result

    def _disc_step(self, real_batch):
        # real_batch shape: [N, C, 512, 512]
        # pass this batch via the discriminator
        real_pred = self.discriminator(real_batch)
        # labels against real batch
        real_labels = torch.ones_like(real_pred).to(self.device)

        # generate the fake samples
        batch_size = len(real_batch)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake_batch = self(noise)
        fake_pred = self.discriminator(fake_batch)
        # labels against fake batch
        fake_labels = torch.zeros_like(fake_pred).to(self.device)

        # calc. the loss
        loss_real = self.loss_adv(real_pred, real_labels)
        loss_fake = self.loss_adv(fake_pred, fake_labels)

        # total discriminator loss
        return (loss_real + loss_fake) / 2.0

    def _get_noise(self, n_samples: int, latent_dim: int):
        return torch.randn(n_samples, latent_dim, device=self.device)



    def _gen_step(self, real_batch):
        # get sample from z
        batch_size = len(real_batch)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake_samples = self(noise)
        # pass via discriminator
        fake_pred = self.discriminator(fake_samples)
        # labels for the generator
        fake_labels = torch.ones_like(fake_pred).to(self.device)
        # calc. the loss
        loss = self.loss_adv(fake_pred, fake_labels)
        return loss

    def on_train_epoch_end(self):
        z = self._get_noise(self.hparams.val_samples, self.hparams.latent_dim).to(self.device)
        # log sampled images
        sample_imgs = self(z)
        grid = to_pil_image(make_grid( (sample_imgs + 1.0 ) / 2.0))
        grid.save(f"{self.hparams.ckpts_dir}/generated_images_{self.global_step}.png")
        self.logger.log_image('generated_images', images=[grid]
                              )



conf = OmegaConf.load("models/cfg.yaml")
model = multi_gan(**conf)

# wandb logger
logger = pl.loggers.wandb.WandbLogger(project="multi_gan",
                                      log_model=False,
                                      config=conf,

                                      )
logger.watch(model, log='gradients', log_freq=10)
# model ckpt callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"{conf.ckpts_dir}",
                                                   filename="multi_gan-{epoch:02d}-{loss_generator:.4f}-{loss_discriminator:.4f}",
                                                   every_n_epochs=10,
                                                   save_last=True,
                                                   verbose=True,
                                                   # save_top_k=10
                                                   )

trainer = pl.Trainer(accelerator='auto', devices='auto', max_epochs=conf.epochs, precision=16, logger=logger, callbacks=[checkpoint_callback],
                     enable_checkpointing=True)

trainer.fit(model)

#%%

from PIL import Image
import os

def make_gif(input_folder, output_file, duration=500):
    # Read all PNG images from the input folder
    images = []
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".png"):
            file_path = os.path.join(input_folder, file_name)
            img = Image.open(file_path)
            images.append(img)

    # Save images as a GIF
    images[0].save(output_file, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

# Example usage:
input_folder = conf.ckpts_dir
output_file = os.path.join(conf.ckpts_dir, "generations.gif")
make_gif(input_folder, output_file)