# Pet-ID

반려견 견종, 털길이, 몸무게 Multi-Label Classification model

모델 결과값 정리 - https://docs.google.com/spreadsheets/d/10T0Qatuifek33ecvgdPP2zkReyDR-7_GtcEvGVkF0pA/edit?usp=sharing

pretrained tflite model - https://drive.google.com/file/d/1NQefzoXkA7mor20PQ5qPt78aB2k9VAPV/view?usp=sharing

## model
mobilenet_v3_small ~ large

reference - https://arxiv.org/abs/1905.02244


## Datasets
AIHUB - https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71520


## Install

1. #### Datasets Downloads

2. #### package install
```commandline
pip install -r requirements.txt
```

3. #### data processing
```commandline
python processing.py
## train_B.csv, valid_B.csv
```

## Start

```commandline
python train.py
```

4. #### torch to onnx
```commandline
python torch_to_onnx.py
```

5. #### onnx to tflite
```commandline
onnx2tf -i dogs14.onnx
```

6. #### metadata add
```commandline
python tflite_metadata_multilabel.py
```