import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


def crop_and_concat(upsampled, bypass):
    _, _, h, w = upsampled.size()
    bypass_cropped = center_crop(bypass, [h, w])
    return torch.cat((upsampled, bypass_cropped), dim=1)


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(PartialConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)  # Initialize mask conv weights to 1
        self.mask_conv.weight.requires_grad = False  # Freeze mask conv weights

    def forward(self, x, mask):
        # Convolution on input
        img_out = self.input_conv(x * mask)

        # Update mask
        with torch.no_grad():
            mask_out = self.mask_conv(mask)
            mask_out = torch.clamp(mask_out, 0, 1)
        
        mask_ratio = self.kernel_size**2 / (mask_out + 1e-8)
        mask_ratio = mask_ratio * mask_out

        img_out = img_out * mask_out  # Ensure the output is zero where mask is zero

        return img_out, mask_out
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.partial_conv1 = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.partial_conv2 = PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.partial_conv1(x, mask)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x, mask = self.partial_conv2(x, mask)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        return x, mask
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decoder = ConvBlock(in_channels, out_channels)
        self.convert_channel = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, img_in, mask_in, img_enc, mask_enc):
        dec = F.interpolate(img_in, scale_factor=2, mode='nearest')
        dec = self.convert_channel(dec)
        dec = crop_and_concat(dec, img_enc)
        d_mask = F.interpolate(mask_in, scale_factor=2, mode='nearest')
        d_mask = self.convert_channel(d_mask)
        d_mask = crop_and_concat(d_mask, mask_enc)
        dec, d_mask = self.decoder(dec, d_mask)
        return dec, d_mask


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, start_dim=64):
        super(UNet, self).__init__()
        # Encoder layers
        self.encoder1 = ConvBlock(in_channels, start_dim)
        self.encoder2 = ConvBlock(start_dim, start_dim * 2)
        self.encoder3 = ConvBlock(start_dim * 2, start_dim * 4)
        self.encoder4 = ConvBlock(start_dim * 4, start_dim * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(start_dim * 8, start_dim * 16)

        # Decoder layers
        self.decoder4 = DecoderBlock(start_dim * 16, start_dim * 8)
        self.decoder3 = DecoderBlock(start_dim * 8, start_dim * 4)
        self.decoder2 = DecoderBlock(start_dim * 4, start_dim * 2)
        self.decoder1 = DecoderBlock(start_dim * 2, start_dim)

        # Final output layer
        self.final = PartialConv2d(start_dim, out_channels, kernel_size=1)

    def forward(self, x, mask):
        # Encoder
        enc1, e_mask1 = self.encoder1(x, mask)
        enc2, e_mask2 = self.encoder2(F.max_pool2d(enc1, 2), F.max_pool2d(e_mask1, 2))
        enc3, e_mask3 = self.encoder3(F.max_pool2d(enc2, 2), F.max_pool2d(e_mask2, 2))
        enc4, e_mask4 = self.encoder4(F.max_pool2d(enc3, 2), F.max_pool2d(e_mask3, 2))

        # Bottleneck
        bottleneck, mask_bottleneck = self.bottleneck(F.max_pool2d(enc4, 2), F.max_pool2d(e_mask4, 2))

        # Decoder
        dec4, d_mask4 = self.decoder4(bottleneck, mask_bottleneck, enc4, e_mask4)
        dec3, d_mask3 = self.decoder3(dec4, d_mask4, enc3, e_mask3)
        dec2, d_mask2 = self.decoder2(dec3, d_mask3, enc2, e_mask2)
        dec1, d_mask1 = self.decoder1(dec2, d_mask2, enc1, e_mask1)

        # Final output
        output, mask_output = self.final(dec1, d_mask1)
        return output