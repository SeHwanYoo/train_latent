# train_ldm_ddp.py
"""
DDP + 클래스 조건(latent concat) Latent-Diffusion 학습 스크립트
- 학습·검증 샘플 목록을 text 파일로 지정.  각 줄:  <class>/<relative_path>
- 모든 GPU에서 학습, rank-0만 checkpoint/샘플 저장
- 1 epoch마다 ckpt, 5 epoch마다 test loss + 샘플(클래스별 1장) 저장
"""

import os, math, time, yaml, random
from pathlib import Path
import torch, torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import yaml

config = yaml.safe_load(open('configs/ddpm.yaml', 'r'))

# ──────────────────────────── ① USER VARIABLES ────────────────────────────
ROOT_DIR      = config['dataset']['train_dir']              # 이미지 최상위 경로
TRAIN_TXT     = config['dataset']['txt_dir'] # 일부만 사용 ← 각 줄 "cls/relpath"
TEST_TXT      = config['dataset']['txt_dir']
CFG_PATH      = './configs/ddpm.yaml'
VQ_CKPT       = config['model']['params']['first_stage_config']['ckpt_path']  
OUT_DIR       = config['log_dir']
EPOCHS        = 100
BATCH         = 16
LR            = 2e-4
IMG_SIZE      = 256
COND_KEY      = 'concat'                     # 클래스 one-hot concat
LOG_EVERY_E   = 5                            # 시각화 주기
# ───────────────────────────────────────────────────────────────────────────

# ──────────────────────── DDP 초기화 & rank util ──────────────────────────
def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, world_size
def is_main(): return dist.get_rank() == 0
def all_barrier(): dist.barrier()

# ───────────────────────────── 데이터셋 ────────────────────────────────────
class SubsetListDataset(Dataset):
    def __init__(self, txt_path, root, img_size):
        self.samples = []
        self.cls2idx = {}
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                cls, rel = line.split('/', 1)
                if cls not in self.cls2idx:
                    self.cls2idx[cls] = len(self.cls2idx)
                self.samples.append( (root+'/'+line, self.cls2idx[cls]) )
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ])
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return {'image': self.tf(img), 'label': torch.tensor(label)}

def make_loader(txt):
    ds = SubsetListDataset(txt, ROOT_DIR, IMG_SIZE)
    sampler = DistributedSampler(ds, shuffle=True)
    loader  = DataLoader(ds, BATCH, sampler=sampler, num_workers=4, pin_memory=True)
    return loader, ds.cls2idx

# ──────────────────────── 모델 로딩 유틸 ───────────────────────────────────
from core.ldm.util import instantiate_from_config
from core.ldm.models.diffusion.ddpm import LatentDiffusion

def load_cfg(yaml_path):
    with open(yaml_path,'r') as fp: return yaml.safe_load(fp)

def build_model(num_classes, device):
    cfg = load_cfg(CFG_PATH)
    cfg['model']['params']['conditioning_key'] = COND_KEY

    # ─ first-stage (VQ-VAE) freeze
    fs_cfg = cfg['model']['params']['first_stage_config']
    first_stage = instantiate_from_config(fs_cfg)
    sd = torch.load(VQ_CKPT, map_location='cpu')
    first_stage.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd, strict=False)
    first_stage.eval().requires_grad_(False)

    # ─ latent diffusion
    model = instantiate_from_config(cfg['model']).to(device)
    model.first_stage_model = first_stage
    model.model.conditioning_key = COND_KEY
    return model

# ─────────────────────────── 보조 함수 ────────────────────────────────────
@torch.no_grad()
def one_hot(y, n_cls, h, w):
    o = torch.zeros(y.size(0), n_cls, h, w, device=y.device)
    o[torch.arange(y.size(0)), y] = 1.
    return o

@torch.no_grad()
def save_samples(model, n_cls, epoch, outdir):
    model.eval()
    y = torch.arange(n_cls, device=next(model.parameters()).device)
    z_shape = (n_cls, model.channels, model.image_size//8, model.image_size//8)
    cond = one_hot(y, n_cls, z_shape[2], z_shape[3])
    z, _ = model.sample(cond=cond, batch_size=n_cls, return_intermediates=True, verbose=False)
    imgs = model.decode_first_stage(z)
    save_image(imgs*0.5+0.5, f'{outdir}/samples_ep{epoch:04d}.png', nrow=int(math.sqrt(n_cls)))
    model.train()

def save_ckpt(model_ddp,opt,epoch,outdir):
    if is_main():
        torch.save({'epoch':epoch,
                    'model':model_ddp.module.state_dict(),
                    'opt':opt.state_dict()},
                   f'{outdir}/ckpt_ep{epoch:04d}.pt')

@torch.no_grad()
def eval_loss(model_ddp, loader, n_cls):
    model_ddp.eval(); tot=n=0
    for b in loader:
        x = b['image'].cuda(non_blocking=True)
        y = b['label'].cuda(non_blocking=True)
        post = model_ddp.module.encode_first_stage(x)
        z = model_ddp.module.get_first_stage_encoding(post).detach()
        cond = one_hot(y,n_cls,z.shape[2],z.shape[3])
        t = torch.randint(0, model_ddp.module.num_timesteps, (z.size(0),), device=z.device).long()
        loss,_ = model_ddp.module.p_losses(z,cond,t)
        tot += loss.item()*z.size(0); n += z.size(0)
    model_ddp.train()
    return tot/n

# ─────────────────────────────── main ─────────────────────────────────────
def main():
    rank, world = ddp_setup()
    if is_main(): os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, train_cls = make_loader(TRAIN_TXT)
    test_loader , _         = make_loader(TEST_TXT)
    NUM_CLASSES = len(train_cls)

    device = torch.device(f'cuda:{rank}')
    model  = build_model(NUM_CLASSES, device)
    model_ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.AdamW(model_ddp.parameters(), lr=LR)

    for ep in range(EPOCHS):
        train_loader.sampler.set_epoch(ep)
        start = time.time()
        for batch in train_loader:
            x = batch['image'].cuda(non_blocking=True)
            y = batch['label'].cuda(non_blocking=True)

            post = model_ddp.module.encode_first_stage(x)
            z = model_ddp.module.get_first_stage_encoding(post).detach()
            cond = one_hot(y, NUM_CLASSES, z.shape[2], z.shape[3])
            t = torch.randint(0, model_ddp.module.num_timesteps, (z.size(0),), device=device).long()
            loss,_ = model_ddp.module.p_losses(z,cond,t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model_ddp.parameters(), 1.0)
            opt.step()

        save_ckpt(model_ddp,opt,ep,OUT_DIR)

        if (ep+1)%LOG_EVERY_E==0 and is_main():
            test_l = eval_loss(model_ddp, test_loader, NUM_CLASSES)
            print(f'[ep {ep}] test loss {test_l:.4f}')
            save_samples(model_ddp.module, NUM_CLASSES, ep, OUT_DIR)

        if is_main():
            print(f'Epoch {ep} done  (loss {loss.item():.4f})  [{time.time()-start:.1f}s]')

    all_barrier()
    if is_main(): print('✓ Training finished')

if __name__ == "__main__":
    main()
