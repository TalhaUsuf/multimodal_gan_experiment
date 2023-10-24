from .discriminator import Discriminator
from .generator import Generator
import torch

def weights_init_normal(m):
    """
    Initialize mean 0 and mean 1 the conv and batch norm layers respectively

    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def build_model(opt):
    """
    build function for getting discriminator and generator models
    :param opt: ArgumentParser
    :return: torch.nn.Module, torch.nn.Module
    """
    discriminator = Discriminator(opt)
    discriminator.apply(weights_init_normal)

    generator = Generator(opt)
    generator.apply(weights_init_normal)

    return discriminator, generator