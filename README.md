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

1. Datasets Downloads

2. package install
```bash
pip install -r requirements.txt
```

3. data processing
```bash
python processing.py

# train_B.csv, valid_B.csv
```

## Start

```bash
python train.py
```
