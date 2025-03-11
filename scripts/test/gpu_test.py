import torch

# CUDA 사용 가능 여부 확인
print(f"CUDA Available: {torch.cuda.is_available()}")

# 현재 사용 가능한 GPU 개수 확인
print(f"GPU Count: {torch.cuda.device_count()}")

# 현재 사용 중인 GPU 정보 확인
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")