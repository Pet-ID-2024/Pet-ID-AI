import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from model_mobilev4 import MultiOutputModel
from dataset import FashionDataset, AttributesDataset, mean, std
import torchvision.transforms as transforms

device = 0
device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")

attributes = AttributesDataset('./train_B.csv')

multioutput_model = MultiOutputModel(
    n_breed_classes=attributes.num_breed,
    n_hair_classes=attributes.num_hair,
    n_weight_classes=attributes.num_weight,
    n_color_classes=attributes.num_color
).to(device)

# 미리 학습된 가중치를 읽어옵니다
model_url = '/home/divus/leeys/dogs/pet_id/checkpoints/2024-12-23_15-16/checkpoint-000045.pth'
batch_size = 1    # 임의의 수

# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
multioutput_model.load_state_dict(torch.load(model_url, map_location=device))

# 모델을 추론 모드로 전환합니다
multioutput_model.eval()

x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = multioutput_model(x)

# 모델 변환
torch.onnx.export(multioutput_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "dogs14.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})