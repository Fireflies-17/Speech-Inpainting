import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from src.utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    # This ResBlock1 class is a fundamental building block often used in deep learning
    # architectures, especially in tasks like speech synthesis or audio processing. It
    # allows the network to capture and propagate relevant features through the use of
    # residual connections. The Leaky ReLU activations introduce non-linearity, and
    # weight normalization helps stabilize and accelerate the training process.
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        # Initialize the residual block with dilated convolutions
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)
        # Additional set of convolutional layers
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        # Forward pass through the residual block
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x  # Residual connection
        return x

    def remove_weight_norm(self):
        # Remove weight normalization for cleanup
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    # Simpler architecture with two convolutional layers and two dilation rates.
    # Compare to ResBlock1, this potentially captures less complex patterns
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    # The Generator class defines a generator model with an architecture consisting of
    # convolutional layers, upsampling layers, and residual blocks. It leverages weight
    # normalization for stable training and provides a method (remove_weight_norm) to
    # remove weight normalization from all layers for inference purposes. The forward
    # pass incorporates leaky ReLU activations and residual connections to generate the
    # final output. The structure and configuration of the generator can be adjusted
    # through the hyperparameters (h).
    def __init__(self, h):
        super(Generator, self).__init__()
        # h: configuration parameters (hyperparameters)
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # Initial convolutional layer with weight normalization.
        self.conv_pre = weight_norm(
            Conv1d(
                getattr(h, "model_in_dim", 128),
                h.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )  # diff: `80` -> `getattr(h, "model_in_dim", 128)`
        # Residual block type (either ResBlock1 or ResBlock2 based on the configuration)
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    # The DiscriminatorP class defines a discriminator model specifically designed for
    # processing periodic signals. It uses a series of convolutional layers to capture
    # hierarchical features of the input signal. The periodic nature of the signal is
    # accounted for by reshaping it into 2D segments. The class provides both the
    # discriminator output and a list of intermediate feature maps, which can be useful
    # for various applications. The architecture allows for flexibility in choosing
    # between weight normalization and spectral normalization based on the
    # use_spectral_norm flag.
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        # period: Number of time steps in each period of the periodic signal
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # Convert the 1D signal to a 2D representation by reshaping it into segments of
        # length period
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        # Process the 2D signal through a series of convolutional layers (convs), each
        # followed by leaky ReLU activation. Feature maps are stored at each layer.
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        # Pass the processed 2D signal through the post-processing convolutional layer
        # (conv_post) with leaky ReLU activation.
        x = self.conv_post(x)
        fmap.append(x)
        # Flatten the output tensor to a 1D representation
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    # The MultiPeriodDiscriminator class acts as an ensemble of discriminators, each
    # specializing in recognizing signals with different periodicities. This
    # architecture is particularly useful for handling signals with varying structures
    # or periodic patterns. During the forward pass, the class computes discriminator
    # outputs and intermediate feature maps for both real and generated signals,
    # providing rich information for training and evaluation.
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    # The DiscriminatorS class is designed to discriminate signals based on their
    # spectrogram representations. It uses a series of convolutional layers to capture
    # hierarchical features in the input signal. The model's architecture allows it to
    # learn complex patterns and structures present in the spectrogram, making it
    # suitable for tasks like signal classification or generation.
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    # The MultiScaleDiscriminator class implements a discriminator model that operates
    # at multiple scales. This is achieved by using multiple instances of the
    # DiscriminatorS class, each focusing on a different level of signal resolution.
    # The average pooling layers help process signals at different resolutions,
    # enabling the model to capture both fine and coarse details in the input signals.
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    """The feature_loss function measures the discrepancy between the feature
    representations of real and generated sources. It quantifies how well the
    generator reproduces the intermediate features observed in real data. The mean
    absolute difference is used as the element-wise loss metric between corresponding
    feature maps. The total loss is then scaled by a factor of 2.

    Args:
        fmap_r (_type_): _description_
        fmap_g (_type_): _description_

    Returns:
        _type_: _description_
    """

    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """The discriminator_loss function computes the adversarial loss for both real and
    generated samples based on the outputs of the discriminator. It uses the mean
    squared difference from the target values (1 for real samples and 0 for generated
    samples) as the adversarial loss metric. The total loss is the sum of adversarial
    losses for real and generated samples. Individual losses are also stored in
    separate lists.

    Args:
        disc_real_outputs (_type_): _description_
        disc_generated_outputs (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    r_losses = []
    g_losses = []
    # Iterating over discriminator outputs for real and generated samples
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # Adversarial loss for real samples
        r_loss = torch.mean((1 - dr) ** 2)
        # Adversarial loss for generated samples
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """The generator_loss function computes the adversarial loss for the generator
    based on the discriminator outputs for generated samples. It uses the mean squared
    difference from the target value (1) as the adversarial loss metric. The total loss
    is the sum of adversarial losses for all generated samples, and individual losses
    are also stored in a list.

    Args:
        disc_outputs (_type_): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    gen_losses = []
    # Iterating over discriminator outputs for generated samples
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
