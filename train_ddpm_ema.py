import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast
# from torch.utils.tensorboard import SummaryWriter

# save_image 
from torchvision.utils import save_image

import yaml
import logging

# 가정: 아래 모듈들은 현재 프로젝트 구조에 맞게 존재합니다.
from core.ldm.util import instantiate_from_config
# from vae import AutoencoderKL
from ddpm import LatentDiffusion
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

def save_checkpoint(logger, ckpt_path, model, optimizer, epoch, global_step, best_loss):
    """체크포인트를 저장합니다."""
    # DDP 모델의 state_dict를 저장할 때는 .module을 사용해야 합니다.
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict(),
        # 'opt_ae_state_dict': opt_ae.state_dict(),
        # 'opt_disc_state_dict': opt_disc.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(state, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")

def load_checkpoint(logger, ckpt_path, model, optimizer, device):
    """체크포인트를 로드합니다."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.module.load_state_dict(ckpt['model_state_dict'])
    # opt_ae.load_state_dict(ckpt['opt_ae_state_dict'])
    # opt_disc.load_state_dict(ckpt['opt_disc_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict']) 
    start_epoch = ckpt['epoch'] + 1
    best_loss = ckpt['best_loss']
    global_step = ckpt['global_step']
    logger.info(f"Checkpoint loaded from {ckpt_path}. Resuming from epoch {start_epoch}.")
    return start_epoch, best_loss, global_step

def main():
    """메인 학습 로직"""
    config = yaml.safe_load(open('configs/ddpm_ema.yaml', 'r'))
    
    # DDP 설정 및 rank 정보 가져오기
    rank, local_rank, world_size = ddp_setup()
    device = local_rank
    
    # writer = None
    # if rank == 0:
        # writer = SummaryWriter(log_dir=config['log_dir'])

    logger = setup_logging(config['log_dir'], rank)
    
    # 모델, 손실 함수, 옵티마이저 초기화
    model_params = config['model']['params']
    model = LatentDiffusion(
        first_stage_config=model_params['first_stage_config'],
        cond_stage_config=model_params['cond_stage_config'],
        unet_config=model_params['unet_config'],
        num_timesteps_cond=model_params['num_timesteps_cond'],
        cond_stage_key=model_params['cond_stage_key'],
        cond_stage_trainable=model_params['cond_stage_trainable'],
        # concat_mode=model_params['concat_mode'],
        # cond_stage_forward=model_params['cond_stage_forward'],
        conditioning_key=model_params['conditioning_key'],
        # scale_factor=model_params['scale_factor'],
        # scale_by_std=model_params['scale_by_std'],
        
        linear_start=model_params['linear_start'],
        linear_end=model_params['linear_end'],
        # cosine_s=model_params['cosine_s'],
        # given_betas=model_params['given_betas'],
        channels=model_params['channels'],
        image_size=model_params['image_size'],
        
        log_every_t=model_params['log_every_t'],
        
        # use_ema = True
        
    ).to(device)
    
    # loss_fn = instantiate_from_config(model_params['lossconfig']).to(device)
    # opt_ae = torch.optim.Adam(list(model.encoder.parameters()) +
    #                           list(model.decoder.parameters()) +
    #                           list(model.quant_conv.parameters()) +
    #                           list(model.post_quant_conv.parameters()),
    #                           lr=config['model']['base_learning_rate'], betas=(0.5, 0.9))
    
    # opt_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
    #                             lr=config['model']['base_learning_rate'], betas=(0.5, 0.9))
    
    # 모델을 DDP로 래핑
    model = DDP(model, device_ids=[device])
    
    # optimizer = torch.optim.AdamW(model.model.parameters(), lr=config['model']['base_learning_rate'])
    inner = model.module
    optimizer = torch.optim.AdamW(inner.model.parameters(),
                                lr=config['model']['base_learning_rate'])
    
    # 데이터셋 및 데이터로더 설정
    train_dataset = ImageDataset(config['dataset']['train_dir'],
                                 config['dataset']['txt_dir'],
                                 config['dataset']['resolution'], 
                                 is_label=True
                                 ) 
    test_dataset = ImageDataset(config['dataset']['train_dir'],
                                config['dataset']['txt_dir'],
                                config['dataset']['resolution'], 
                                is_label=True
                                )
    
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
        start_epoch, best_loss, global_step = load_checkpoint(logger, last_ckpt_path, model, optimizer, device)
    
    os.makedirs(config['model']['samples_dir'], exist_ok=True)
    
    patience = 10 
    epochs_no_improve = 0
        
    # 학습 루프
    for epoch in range(start_epoch, config['model']['epochs']):
        model.train()
        train_sampler.set_epoch(epoch) # 매 에폭마다 샘플러의 시드를 변경하여 데이터 셔플링
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                encoder_posterior = model.module.encode_first_stage(images)
                z = model.module.get_first_stage_encoding(encoder_posterior).detach()
                
            # c = model.module.get_learned_conditioning(labels)
            # c = model.module.get_learned_conditioning({"class_label": labels.long()})
            
            optimizer.zero_grad()
            # loss, log_dict = model(z, c) 
            loss, log_dict = model(z, {"class_label": labels.long()}) 
            loss.backward() 
            optimizer.step() 

            if rank == 0 and i % 100 == 0:
                logger.info(f"Epoch: {epoch} | Batch: {i}/{len(train_loader)} | Loss: {loss.item():.6f}")
                # for key, val in log_dict.items():
                #     writer.add_scalar(key, val, global_step)
                for key, val in log_dict.items():
                    logger.info(f"{key}: {val:.6f}")
            
            global_step += 1
            
        torch.distributed.barrier()
        
        if rank == 0:
            model.eval() # 평가 모드로 변경

            # --- 1. 샘플 이미지 저장 (매 에폭마다 수행) ---
            with torch.no_grad():
                # 검증 데이터셋에서 고정된 샘플 배치를 가져옵니다.
                # val_batch = next(iter(val_loader))
                # val_images = val_batch["image"].to(device)
                # val_labels = val_batch["class_label"].to(device)
                for images, labels in val_loader:
                    val_images = images.to(device)
                    val_labels = labels.to(device)
                    break

                # VAE를 통해 원본 이미지의 잠재 벡터 z를 얻습니다.
                encoder_posterior = model.module.encode_first_stage(val_images)
                z = model.module.get_first_stage_encoding(encoder_posterior)
                
                # 컨디션 c를 준비합니다.
                # c = model.module.get_learned_conditioning(val_labels)
                
                # LDM 샘플링을 통해 잠재 벡터 복원
                # (모델에 `sample` 메서드가 구현되어 있다고 가정)
                # samples_z, _ = model.module.sample(cond={"class_label": val_labels.long()}, batch_size=val_images.shape[0], return_intermediates=True)
                
                # cond_emb = model.module.get_learned_conditioning(val_labels.long())
                cond_dict  = {"class_label": val_labels.long()}          # ① 딕셔너리로 래핑
                cond_emb   = model.module.get_learned_conditioning(cond_dict) 
                samples_z, _ = model.module.sample(
                    cond=cond_emb,                       # 🔸 딕셔너리 NO
                    batch_size=val_images.size(0),
                    return_intermediates=True)
                
                # 복원된 잠재 벡터를 VAE 디코더로 이미지화
                x_samples = model.module.decode_first_stage(samples_z)

                # 원본 이미지와 생성된 이미지를 비교하기 위해 저장
                comparison_grid = torch.cat([val_images[:8], x_samples[:8]])
                save_image(comparison_grid, os.path.join(config['model']['samples_dir'], f"epoch_{epoch:04d}.png"), nrow=8, normalize=True, value_range=(-1, 1))
                logger.info(f"Epoch {epoch:03d} | Sample images saved.")

            # --- 2. 검증 및 최고 모델 저장 (5 에폭마다 수행) ---
            # if (epoch + 1) % 5 == 0:
            #     total_val_loss = 0.0
            #     with torch.no_grad():
            #         # for batch in val_loader:
            #         #     images = batch["image"].to(device)
            #         #     labels = batch["class_label"].to(device)
            #         for images, labels in val_loader:
            #             images = images.to(device)
            #             labels = labels.to(device)
                        
            #             encoder_posterior = model.module.encode_first_stage(images)
            #             z = model.module.get_first_stage_encoding(encoder_posterior)
            #             c = model.module.get_learned_conditioning(labels)
                        
            #             loss, _ = model(z, c)
            #             total_val_loss += loss.item()

            #     avg_val_loss = total_val_loss / len(val_loader)
            #     logger.info(f"Epoch {epoch:03d} | --- Validation --- | Avg Loss: {avg_val_loss:.6f}")
                # writer.add_scalar("validation/avg_loss", avg_val_loss, epoch)
            if (epoch + 1) % 5 == 0:
                total = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:          # ← tuple
                        images = images.to(device)
                        labels = labels.to(device)

                        z = model.module.get_first_stage_encoding(
                                model.module.encode_first_stage(images))
                        cond = {"class_label": labels.long()}
                        loss, _ = model(z, cond)
                        total += loss.item()

                avg_val_loss = total / len(val_loader)
                logger.info(f"Epoch {epoch:03d} | val/avg_loss: {avg_val_loss:.6f}")

                # 최고 성능 모델 저장
                # is_best = avg_val_loss < best_loss
                # if is_best:
                #     best_loss = avg_val_loss
                #     logger.info(f"New best validation loss: {best_loss:.6f}. Saving best model...")
                #     save_checkpoint(logger, os.path.join(ckpt_dir, "best.pth"), model, optimizer, epoch, global_step, best_loss)
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    epochs_no_improve = 0 # 카운터 리셋
                    logger.info(f"New best validation loss: {best_loss:.6f}. Saving best model...")
                    
                    save_checkpoint(logger, os.path.join(ckpt_dir, "best.pth"), model, optimizer, epoch, global_step, best_loss)
                else:
                    epochs_no_improve += 1
                    logger.info(f"Validation loss did not improve for {epochs_no_improve} validation runs.")
            

            # --- 3. 최신 상태 체크포인트 저장 (매 에폭마다 수행) ---
            save_checkpoint(logger, os.path.join(ckpt_dir, "last.pth"), model, optimizer, epoch, global_step, best_loss)

        stop_signal = torch.tensor(0.0, device=device)
        if rank == 0 and epochs_no_improve >= patience:
            stop_signal.fill_(1.0)
            logger.info(f"Early stopping triggered after {patience} validation runs without improvement.")
        
        # 신호를 모든 프로세스에 브로드캐스트
        broadcast(stop_signal, src=0)
        
        # 신호를 받으면 루프 중단
        if stop_signal.item() == 1:
            break

        torch.distributed.barrier()
        
        # 다음 에폭을 위해 모든 프로세스가 동기화될 때까지 대기
        torch.distributed.barrier()
        
    # if rank == 0:
    #     writer.close()

    ddp_cleanup()

if __name__ == '__main__':
    main()