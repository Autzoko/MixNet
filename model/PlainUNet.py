import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class HybridUNet(nn.Module):
    """
    Hybrid U-Net Segmentation Model
    Combines CNN + Mamba + Window Attention Encoder with U-Net Decoder.
    """

    def __init__(
            self,
            in_ch: int = 1,
            num_classes: int = 1,
            base_ch: int = 32,
            encoder_kwargs: Optional[dict] = None,
            decoder_kwargs: Optional[dict] = None
    ):
        super().__init__()

        self.in_ch = in_ch
        self.num_classes = num_classes
        self.base_ch = base_ch

        encoder_config = {
            'in_ch': in_ch,
            'base_ch': base_ch
        }

        if encoder_kwargs is not None:
            encoder_config.update(encoder_kwargs)
        
        try:
            from model.pipeline.encoder.plain_encoder import PlainCNNEncoder
        except ImportError:
            raise ImportError("Could not import HybridEncoder. Please ensure the model.pipeline.encoder.mixed_encoder module is available.")
        
        self.encoder = PlainCNNEncoder(**encoder_config)
        enc_channels = (base_ch, base_ch * 2, base_ch * 4)

        decoder_config = {
            "enc_channels": enc_channels,
            "dec_channels": (base_ch * 4, base_ch * 2, base_ch),
            "num_classes": num_classes,
            'up_mode': 'bilinear',
            'align_corners': False,
            'norm_type': 'batch',
            'use_bottleneck': True
        }

        if decoder_kwargs is not None:
            decoder_config.update(decoder_kwargs)

        try:
            from model.pipeline.decoder.unet_decoder import UNetDecoder2D
        except ImportError:
            raise ImportError("Could not import UNetDecoder2D. Please ensure the model.pipeline.decoder.unet_decoder module is available.")
        
        self.decoder = UNetDecoder2D(**decoder_config)

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

    def forward(
            self,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Feed forward function.

        Args:
            - x: (B, in_ch, H, W) - input image tensor
        Returns:
            - logits: (B, num_classes, H, W) - segmentation logits
            - aux_outputs: dict containing auxiliary outputs from the encoder
                - features: [f1, f2, f3] - multi-scale features
                - global: (B, D) - global contrastive embedding
                - local: (B, N_patches, D) or None - local contrastive embedding
                - "x3", "x2", "x1": intermediate decoder features if return_intermediate=True
        """
        features, global_repr, local_repr = self.encoder(x)

        logits, intermediates = self.decoder(features, return_intermediate=True)

        aux_outputs = {
            "features": features,
            "global": global_repr,
            "local": local_repr,
            "x3": intermediates['x3'],
            "x2": intermediates['x2'],
            "x1": intermediates['x1']
        }

        return logits, aux_outputs

    def get_encoder(self) -> nn.Module:
        """ Returns the encoder module. """
        return self.encoder
    def get_decoder(self) -> nn.Module:
        """ Returns the decoder module. """
        return self.decoder
    
    def freeze_encoder(self):
        """ Freezes the encoder parameters. """
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters have been frozen.")

    def unfreeze_encoder(self):
        """ Unfreezes the encoder parameters. """
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder parameters have been unfrozen.")

    def freeze_decoder(self):
        """ Freezes the decoder parameters. """
        for param in self.decoder.parameters():
            param.requires_grad = False
        print("Decoder parameters have been frozen.")

    def unfreeze_decoder(self):
        """ Unfreezes the decoder parameters. """
        for param in self.decoder.parameters():
            param.requires_grad = True
        print("Decoder parameters have been unfrozen.")

    def get_model_size(self) -> Dict[str, float]:
        """ Returns the model size in MB. """
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        return {
            "total_params": total_params,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "model_size_MB": total_params * 4 / 1024 / 1024  # assuming FP32
        }
    
    def print_model_info(self):
        stats = self.get_model_size()

        print("=" * 60)
        print("Hybrid U-Net Model Information")
        print("=" * 60)

        print(f"Input Channels: {self.in_ch}")
        print(f"Output Classes: {self.num_classes}")
        print(f"Base Channels: {self.base_ch}\n")

        print("Encoder Configuration:")
        for k, v in self.encoder_config.items():
            print(f"  {k}: {v}")
        print("\nDecoder Configuration:")
        for k, v in self.decoder_config.items():
            print(f"  {k}: {v}")
        
        print("\nModel Statistics:")
        print(f"  Total Parameters: {stats['total_params']:,}")
        print(f"  Encoder Parameters: {stats['encoder_params']:,}")
        print(f"  Decoder Parameters: {stats['decoder_params']:,}")
        print(f"  Model Size: {stats['model_size_MB']:.2f} MB (FP32)")
        print("=" * 60)

