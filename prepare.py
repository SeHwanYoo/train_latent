# prepare_models.py

import yaml
from core.ldm.util import instantiate_from_config
from vae import AutoencoderKL

# 이 스크립트는 모델과 손실 함수를 딱 한 번 생성해서
# 필요한 파일들을 미리 다운로드하는 역할만 합니다.
def main():
    print("--- 사전 학습 모델 다운로드를 시작합니다 ---")

    # 설정 파일 로드
    config_path = "configs/autoencoder_kl.yaml"
    config = yaml.safe_load(open(config_path, 'r'))
    model_params = config['model']['params']

    # 1. AutoencoderKL 모델 생성 (필요 시)
    # model = AutoencoderKL(ddconfig=model_params['ddconfig'], embed_dim=model_params['embed_dim'])
    # print("AutoencoderKL 모델 구조 생성 완료.")

    # 2. LPIPSWithDiscriminator 손실 함수 생성 (VGG 가중치 다운로드 유발)
    loss_fn = instantiate_from_config(model_params['lossconfig'])
    print("LPIPS 손실 함수 생성 및 VGG 가중치 다운로드 완료.")

    print("--- 모든 다운로드가 완료되었습니다. 이제 DDP 학습을 실행할 수 있습니다. ---")

if __name__ == '__main__':
    main()