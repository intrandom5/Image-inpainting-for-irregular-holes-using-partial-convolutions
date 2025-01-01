from training_utils import log_results, print_epoch_summary, PartialUNetLoss

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim


def forward(data, generator, restore_crit, config, return_outputs=False):
    inputs, targets, mask = data

    # Forward pass
    outputs = generator(inputs, mask)

    valid_loss, hole_loss, perceptual_loss, style_loss, tv_loss = restore_crit(
        outputs, targets, mask
    )

    g_loss = (
        valid_loss + hole_loss + perceptual_loss + style_loss + tv_loss
    )

    losses = [valid_loss, hole_loss, perceptual_loss, 
              style_loss, tv_loss, g_loss]

    batch_ssim = ssim(outputs, targets, data_range=1.0, size_average=True)

    if return_outputs:
        return losses, batch_ssim, outputs

    return losses, batch_ssim

def inpainting_train(train_loader, valid_loader, generator, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = generator.to(device)

    # 손실 함수 및 옵티마이저 정의
    restore_crit = PartialUNetLoss(config.loss_weights, device)

    optimizer_g = optim.Adam(generator.parameters(), lr=config.generator_lr)
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min')

    # TensorBoard 설정
    writer = SummaryWriter(config.log_dir)
    best_ssim = 0
    best_generator = None

    def run_epoch(loader, is_train=True, test_step=50):
        generator.train() if is_train else generator.eval()
        total_ssim = 0.0
        total_losses = [0.0 for _ in range(len(config.loss_weights)+1)]
        example_images = []

        for cnt, data in enumerate(tqdm(loader)):
            data = [d.to(device) for d in data]
            if is_train:
                with torch.amp.autocast(device_type=device):
                    losses, batch_ssim = forward(
                        data, generator, restore_crit, 
                        config, return_outputs=False
                    )
                # Gradient update
                optimizer_g.zero_grad()
                losses[-1].backward() # g_loss
                optimizer_g.step()
            else:
                with torch.no_grad():
                    losses, batch_ssim, outputs = forward(
                        data, generator, restore_crit,
                        config, return_outputs=True
                    )

                # 예제 이미지 저장
                if len(example_images) < 5:
                    example_images.append((data[0][0], outputs[0], data[1][0]))

            total_ssim += batch_ssim.item()
            total_losses = [total_loss+loss.item() for total_loss, loss in zip(total_losses, losses)]

            if test_step > 0 and cnt == test_step:
                break

        # 결과 반환
        step = 50 if config.test else len(loader)
        total_losses = [loss / step for loss in total_losses]
        avg_ssim = total_ssim / step

        return total_losses, avg_ssim, example_images

    # 학습 loop 시작
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")

        # Training
        test_step = 50 if config.test else 0
        train_results = run_epoch(
            train_loader, is_train=True, test_step=test_step)
        valid_results = run_epoch(
            valid_loader, is_train=False, test_step=test_step)

        # Scheduler update
        log_results(writer, train_results, valid_results, epoch)
        print_epoch_summary(epoch, train_results, valid_results)

        scheduler_g.step(valid_results[0][-1])

        # Best model 저장
        if valid_results[1] > best_ssim:  # valid_ssim
            best_ssim = valid_results[1]
            best_generator = generator

    # 모델 저장
    if not config.test:
        torch.save(best_generator.state_dict(), os.path.join(config.log_dir, "best_generator.pt"))
    writer.close()
    print("Training completed!")