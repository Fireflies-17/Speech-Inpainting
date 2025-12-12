import torch
import torch.nn as nn

from src.utils import AttrDict
from src.models import Generator
from src.modules.jukebox import Encoder, Decoder
from src.modules.vq import Bottleneck


class FoVQVAE(nn.Module):
    # The FoVQVAE class represents a variation of a VQ-VAE (Vector Quantized Variational
    # Autoencoder) designed specifically for handling fundamental frequency (fo) data.
    # It consists of an encoder, a vector quantization (VQ) bottleneck, and a decoder.
    # The model is used for training both the encoder and the decoder for fo data.

    def __init__(self, h):
        super().__init__()

        self.encoder = Encoder(**h.f0_encoder_params)
        self.vq = Bottleneck(**h.f0_vq_params)
        self.decoder = Decoder(**h.f0_decoder_params)

    def forward(self, **kwargs):
        """
        Args:
            kwargs
                f0 - Ground-Truth fo
        Returns:
            out_estim - Reconstructed fo
            commit_losses - VQ commitment loss
            metrics - VQ metrics
        """
        # Ground-Truth is ristricted to fo
        in_gt = kwargs["f0"]
        f0_h = self.encoder(in_gt)
        _, f0_h_q, commit_losses, metrics = self.vq(f0_h)
        out_estim = self.decoder(f0_h_q)

        return out_estim, commit_losses, metrics


class CodeGenerator(Generator):
    # The CodeGenerator class is a decoder model that combines an embedding layer with
    # a HiFi-GAN (High Fidelity Generative Adversarial Network) generator. This class is
    # designed for generating waveforms from discrete content, continuous fundamental
    # frequency (fo), and discrete speaker information.

    def __init__(self, h):
        super().__init__(h)

        # Embedding of ContentCodeGenerator
        self.emb_c = nn.Embedding(h.num_embeddings, h.embedding_dim)
        # VQ-VAE Content
        self.code_encoder = None
        self.code_vq = None
        if h.lambda_commit_code:
            self.code_encoder = Encoder(**h.code_encoder_params)
            self.code_vq = Bottleneck(**h.code_vq_params)
            self.emb_c = None
        # Embedding of Pitch
        # Fixed fo Encoder
        self.fo_vqvae = None
        if h.f0_quantizer_path:
            self.emb_p = nn.Embedding(
                h.f0_quantizer["f0_vq_params"]["l_bins"], h.embedding_dim
            )
            self.fo_vqvae = FoVQVAE(AttrDict(h.f0_quantizer))
            self.fo_vqvae.load_state_dict(
                torch.load(h.f0_quantizer_path, map_location="cpu")["generator"]
            )
            self.fo_vqvae.eval()

        # Embedding of Speaker
        # TODO : problem of speaker size with ljspeech dataset
        self.emb_s = nn.Embedding(h.spk_embeddings, h.embedding_dim)
        # NOTE: `num_embeddings=200` is too much. LJSpeech needs just 1, VCTK needs 109.

    @staticmethod
    def _upsample(signal, max_frames):
        """
        Args:
            signal     :: (*) - Target signal
            max_frames :: int - Reference length, until which signal is upsampled
        Returns:
            :: (B, Feat=feat|1, T=max_frames)
        """
        # (B, Feat=feat, T=t) | (B, Feat=feat) | (B,) -> (B, Feat=feat|1, T=t|1)
        # print("Signal", signal)
        # print("Signal", signal.shape)
        # print("Frames", max_frames)
        if signal.dim() == 3:
            pass
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
        else:
            signal = signal.view(-1, 1, 1)
        bsz, channels, length_t = signal.size()
        # print("Bsz", bsz)
        # print("Channels", channels)
        # print("Length_t", length_t)
        # (B, Feat, T) -> (B, Feat, T, 1) -> (B, Feat, T, ref//T)
        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // length_t)
        # print("Signal", signal)
        # print("Signal", signal.shape)

        # pad zeros as needed (if signal's shape does not divide completely with
        # max_frames)
        # (ref - T * (ref//T)) // (ref//T)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        # (B, Feat, T, ref/T) -> (B, Feat, T=ref)
        signal = signal.view(bsz, channels, max_frames)
        # print("Signal", signal)
        # print("Signal", signal.shape)
        return signal

    def forward(self, **kwargs):
        """
        Args:
            kwargs
                code :: (B, Frame)    - Content unit series
                f0   :: (B, 1, Frame) - Fundamental frequency series
                emb  :: (B, Emb)      - speaker embedding
                spkr :: (B, 1)        - speaker index
        Returns:
            wave_estim - Estimated waveform
        """

        # Inputs - discrete Content, continuous fo, discrete Speaker
        z_c = kwargs["code"]
        if self.code_vq and kwargs["code"].dtype is torch.int64:
            emb_c = self.code_vq.level_blocks[0].k[z_c].transpose(1, 2)
        elif self.code_vq:
            code_h = self.code_encoder(z_c)
            _, code_h_q, code_commit_losses, code_metrics = self.code_vq(code_h)
            emb_c = code_h_q[0]
        else:
            fo, emb, z_s = kwargs["f0"], kwargs["emb"], kwargs["spkr"]
            # Embedding :: (B, Frame) -> (B, Frame, Emb) -> (B, Emb, Frame)
            emb_c = self.emb_c(z_c).transpose(1, 2)
            # emb_s = self.emb_s(z_s).transpose(1, 2)
            emb_s = emb
        # fo encoding - In-place fo encoding to z_p by fixed Encoder-VQ
        if self.fo_vqvae:
            self.fo_vqvae.eval()
            assert not self.fo_vqvae.training, "VQ is in training status!!!"
            h_p = [x.detach() for x in self.fo_vqvae.encoder(fo)]
            # z_p :: ()
            z_p = [x.detach() for x in self.fo_vqvae.vq(h_p)[0]][0].detach()
            emb_p = self.emb_p(z_p).transpose(1, 2)
        if self.h.f0_stats:
            # Up↑/Concat :: (B, Emb=emb, Frame) -> (B, Emb=2*emb, Frame=max(c,p))
            if emb_c.shape[-1] < emb_p.shape[-1]:
                # print("Code up")
                emb_c = self._upsample(emb_c, emb_p.shape[-1])
            else:
                # print("Pitch up")
                emb_p = self._upsample(emb_p, emb_c.shape[-1])
            emb_c_p = torch.cat([emb_c, emb_p], dim=1)
        else:
            emb_c_p = emb_c
        if self.h.multispkr:
            # :: (B, Emb=2*emb, Frame=max(c,p)) -> (B, Emb=3*emb, Frame=max(c,p,s))
            # print("Speaker up")
            emb_s = self._upsample(emb_s, emb_c_p.shape[-1])
            emb_c_p_s = torch.cat([emb_c_p, emb_s], dim=1)
        else:
            emb_c_p_s = emb_c_p
        # VQ-VAE Content Encoder
        if self.code_vq:
            for k, feat in kwargs.items():
                if k in ["spkr", "code", "f0"]:
                    continue
                # print("Feat up")
                feat = self._upsample(feat, emb_c.shape[-1])
                emb_c = torch.cat([emb_c, feat], dim=1)
            return (
                super().forward(emb_c),
                code_commit_losses[0],
                code_metrics,
            )
        # HiFi-GAN Generator
        else:
            wave_estim = super().forward(emb_c_p_s)
            return wave_estim
