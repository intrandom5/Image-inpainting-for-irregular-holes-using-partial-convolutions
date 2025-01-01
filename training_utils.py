from torchvision.utils import make_grid
from torchvision.models import vgg16
import torch.nn.functional as F
import torch.nn as nn
import torch

loss_names = [
    "valid_loss", "hole_loss", "perceptual_loss", "style_loss", "tv_loss", "g_loss"
]

def log_results(writer, train_results, valid_results, epoch):
    # === TensorBoard Logging ===
    for i, loss_name in enumerate(loss_names):
        writer.add_scalar('Train/'+loss_name, train_results[0][i], epoch)
    writer.add_scalar('Train/SSIM', train_results[1], epoch)

    for i, loss_name in enumerate(loss_names):
        writer.add_scalar('Validation/'+loss_name, valid_results[0][i], epoch)
    writer.add_scalar('Validation/SSIM', valid_results[1], epoch)

    images_to_log = []
    for inp, out, tgt in valid_results[-1]:
        if inp.shape[0] != tgt.shape[0]:
            inp = inp.repeat(3, 1, 1) # 흑백을 RGP 채널로 변환해 다른 이미지들과 채널을 맞춤.
        images_to_log.extend([inp.cpu(), out.cpu(), tgt.cpu()])  # 입력, 출력, 정답 추가
    grid = make_grid(images_to_log, nrow=3, normalize=True, value_range=(-1, 1))
    writer.add_image(f'Validation Results/Epoch {epoch+1}', grid, epoch)

def print_epoch_summary(epoch, train_results, valid_results):
    print(f"Epoch [{epoch+1}] results:")
    message = "[Train]\n"
    for key, result in zip(loss_names, train_results[0]):
        message += key + " " + f": {result}, "
    print(message[:-2])
    message = "[Valid]\n"
    for key, result in zip(loss_names, valid_results[0]):
        message += key + " " + f": {result}, "
    print(message[:-2])


class PartialUNetLoss(nn.Module):
    def __init__(self, loss_weights, device):
        super().__init__()
        self.vgg = vgg16(weights="IMAGENET1K_V1").features[:17]
        self.vgg = torch.nn.ModuleList(self.vgg)
        self.vgg.to(device)
        if type(loss_weights) == str:
            loss_weights = eval(loss_weights)
        self.loss_weights = loss_weights

    def vgg_pooling(self, x):
        with torch.no_grad():
            pooling_idx = [4, 9, 16]
            pooling_outputs = []
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                if i in pooling_idx:
                    pooling_outputs.append(x)
        return pooling_outputs
    
    def get_comp_img(self, pred_img, target_img, mask):
        comp_img = mask * target_img + (1-mask) * pred_img
        return comp_img
    
    def total_variation_loss(self, mask, y_comp):
        # Dilate the mask
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)  # 3x3 kernel
        if mask.shape[1] == 3:
            mask = mask[:, :1, :, :]
        dilated_mask = F.conv2d(1 - mask, kernel, padding=1)
        
        # Threshold the dilated mask to create binary values
        dilated_mask = (dilated_mask > 0).float()

        # Apply dilated mask to y_comp
        P = dilated_mask * y_comp

        # Calculate total variation loss within the masked area
        dh = P[:, :, 1:, :] - P[:, :, :-1, :]
        dw = P[:, :, :, 1:] - P[:, :, :, :-1]
        tv_loss = torch.sum(torch.abs(dh)) + torch.sum(torch.abs(dw))
        valid_pixels = torch.sum(dilated_mask) + 1e-8
        
        return tv_loss / valid_pixels
    
    def remove_inf_nan(self, x):
        x[x!=x]=0
        x[~torch.isfinite(x)]=0
        return x
    
    def gram_matrix(self, x):
        x = torch.clamp(x, -1e3, 1e3)  # 값 제한
        b, c, w, h = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (c * h * w + 1e-8)
        return self.remove_inf_nan(gram)
    
    def forward(self, pred_img, target_img, mask):
        comp_img = self.get_comp_img(pred_img, target_img, mask)

        # pixel loss
        valid_loss = F.l1_loss(pred_img*mask, target_img*mask)
        hole_loss = F.l1_loss(pred_img*(1-mask), target_img*(1-mask))
        
        # vgg pooling
        tgt_pools = self.vgg_pooling(target_img)
        out_pools = self.vgg_pooling(pred_img)
        comp_pools = self.vgg_pooling(comp_img)

        # perceptual loss & style loss
        perceptual_loss = 0
        style_loss = 0
        for tgt, out, comp in zip(tgt_pools, out_pools, comp_pools):
            perceptual_loss += F.l1_loss(out, tgt)
            perceptual_loss += F.l1_loss(comp, tgt)
            out_gram = self.gram_matrix(out)
            tgt_gram = self.gram_matrix(tgt)
            style_loss += F.l1_loss(out_gram, tgt_gram)

        # total variation loss
        tv_loss = self.total_variation_loss(mask, comp_img)

        total_loss = [self.loss_weights[0]*valid_loss,
                    self.loss_weights[1]*hole_loss,
                    self.loss_weights[2]*perceptual_loss,
                    self.loss_weights[3]*style_loss,
                    self.loss_weights[4]*tv_loss]
        
        return total_loss