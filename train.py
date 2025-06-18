import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# save_image 
from torchvision.utils import save_image

import yaml
import logging

# 가정: 아래 모듈들은 현재 프로젝트 구조에 맞게 존재합니다.
from core.ldm.util import instantiate_from_config
from vae import AutoencoderKL
from datasets import ImageDataset

def setup_logging(log_dir, rank):
    """로깅 설정. rank 0 프로세스에서만 파일 및 콘솔에 로그를 출력합니다."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # rank 0에서만 핸들러를 추가하여 중복 로깅 방지
    if rank == 0 and not logger.handlers:
        log_path = os.path.join(log_dir, "training.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 스트림(콘솔) 핸들러
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    elif rank != 0:
        # 다른 rank의 프로세스는 로그를 출력하지 않도록 Null 핸들러 추가
        logger.addHandler(logging.NullHandler())
    return logger

def ddp_setup():
    """torchrun 환경 변수를 사용하여 분산 프로세스 그룹을 초기화합니다."""
    init_process_group(backend="nccl")
    # torchrun이 설정한 환경 변수에서 rank 정보를 가져옵니다.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def ddp_cleanup():
    """프로세스 그룹을 정리합니다."""
    destroy_process_group()

def save_checkpoint(logger, ckpt_path, model, opt_ae, opt_disc, epoch, global_step, best_loss):
    """체크포인트를 저장합니다."""
    # DDP 모델의 state_dict를 저장할 때는 .module을 사용해야 합니다.
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict(),
        'opt_ae_state_dict': opt_ae.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(state, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")
    
def save_checkpoint_epoch(ckpt_path, model, opt_ae, opt_disc, epoch, global_step, best_loss):
    """체크포인트를 저장합니다."""
    # DDP 모델의 state_dict를 저장할 때는 .module을 사용해야 합니다.
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict(),
        'opt_ae_state_dict': opt_ae.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(state, ckpt_path)
    # logger.info(f"Checkpoint saved to {ckpt_path}")

def load_checkpoint(logger, ckpt_path, model, opt_ae, opt_disc, device):
    """체크포인트를 로드합니다."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.module.load_state_dict(ckpt['model_state_dict'])
    opt_ae.load_state_dict(ckpt['opt_ae_state_dict'])
    opt_disc.load_state_dict(ckpt['opt_disc_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_loss = ckpt['best_loss']
    global_step = ckpt['global_step']
    logger.info(f"Checkpoint loaded from {ckpt_path}. Resuming from epoch {start_epoch}.")
    return start_epoch, best_loss, global_step

def main():
    """메인 학습 로직"""
    config = yaml.safe_load(open('configs/autoencoder_kl.yaml', 'r'))
    
    # DDP 설정 및 rank 정보 가져오기
    rank, local_rank, world_size = ddp_setup()
    device = local_rank

    logger = setup_logging(config['log_dir'], rank)
    
    # 모델, 손실 함수, 옵티마이저 초기화
    model_params = config['model']['params']
    model = AutoencoderKL(
        ddconfig=model_params['ddconfig'],
        embed_dim=model_params['embed_dim'],
    ).to(device)
    
    loss_fn = instantiate_from_config(model_params['lossconfig']).to(device)
    
    opt_ae = torch.optim.Adam(list(model.encoder.parameters()) +
                              list(model.decoder.parameters()) +
                              list(model.quant_conv.parameters()) +
                              list(model.post_quant_conv.parameters()),
                              lr=float(config['model']['base_learning_rate']), betas=(0.5, 0.9))
    
    opt_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                                lr=float(config['model']['base_learning_rate']), betas=(0.5, 0.9))
    
    # 모델을 DDP로 래핑
    model = DDP(model, device_ids=[device])
    
    # 데이터셋 및 데이터로더 설정
    train_dataset = ImageDataset(config['dataset']['train_dir'],
                                 config['dataset']['txt_dir'],
                                 config['dataset']['resolution'])
    test_dataset = ImageDataset(config['dataset']['train_dir'],
                                config['dataset']['txt_dir'],
                                config['dataset']['resolution'])
    
    # DistributedSampler 사용 시, DataLoader의 shuffle은 False여야 합니다.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0
    
    # 체크포인트 로드 (rank 0에서만 실행해도 되지만, 모든 프로세스가 동일한 상태에서 시작하도록 여기서 로드)
    ckpt_dir = config['model']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(last_ckpt_path):
        start_epoch, best_loss, global_step = load_checkpoint(logger, last_ckpt_path, model, opt_ae, opt_disc, device)
    
    os.makedirs(config['model']['samples_dir'], exist_ok=True)
        
    # 학습 루프
    for epoch in range(start_epoch, config['model']['epochs']):
        model.train()
        train_sampler.set_epoch(epoch) # 매 에폭마다 샘플러의 시드를 변경하여 데이터 셔플링
        
        for i, images in enumerate(train_loader):
            images = images.to(device)
            reconstructions, posterior = model(images)
            
            # Generator (Autoencoder) 업데이트
            opt_ae.zero_grad()
            aeloss, log_dict_ae = loss_fn(images, reconstructions, posterior, 0, global_step,
                                          last_layer=model.module.get_last_layer(), split="train")
            aeloss.backward()
            opt_ae.step()
            
            # Discriminator 업데이트
            opt_disc.zero_grad()
            discloss, log_dict_disc = loss_fn(images, reconstructions, posterior, 1, global_step,
                                              last_layer=model.module.get_last_layer(), split="train")
            discloss.backward()
            opt_disc.step()

            if rank == 0 and i % 50 == 0:
                logger.info(f"Epoch: {epoch} | Batch: {i}/{len(train_loader)} | AE Loss: {aeloss.item():.4f} | Disc Loss: {discloss.item():.4f}")
            
            global_step += 1
            
            

        with torch.no_grad():
            x = next(iter(val_loader)).to(device)
            x_hat, _ = model(x, sample_posterior=False)
            # grid = torch.cat([x[:8], x_hat[:8]], dim=0)  # 원본/재구성
            # save_image(grid, f"{config['model']['samples_dir']}/epoch_{epoch}.png", nrow=8, normalize=True, value_range=(-1,1))
            grid = torch.cat([x, x_hat], dim=0)
            save_image(grid, f"{config['model']['samples_dir']}/epoch_{epoch}.png", nrow=x.size(0), normalize=True, value_range=(-1,1))
            
            
        if rank == 0: 
            save_checkpoint_epoch(os.path.join(ckpt_dir, f"epoch_{epoch}.pth"), model, opt_ae, opt_disc, epoch, global_step, best_loss)
        
        if epoch > 3 and rank == 0:
            model.eval()
            total_val_loss = 0.0
            # with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructions, posterior = model(images)
                aeloss, _ = loss_fn(images, reconstructions, posterior, 0, global_step,
                                    last_layer=model.module.get_last_layer(), split="val")
                total_val_loss += aeloss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            logger.info(f"Epoch: {epoch} | Average Validation AE Loss: {avg_val_loss:.4f}")

            # 최신 체크포인트 저장
            # save_checkpoint(logger, ckpt_dir, f"latest_{epoch}.pth", model, opt_ae, opt_disc, epoch, global_step, best_loss)

            # 최고 성능 모델 저장
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                logger.info(f"New best validation loss: {best_loss:.4f}. Saving best model.")
                save_checkpoint(logger, os.path.join(ckpt_dir, "best.pth"), model, opt_ae, opt_disc, epoch, global_step, best_loss)

    ddp_cleanup()

if __name__ == '__main__':
    main()
