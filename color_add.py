""" 
@Author: LYS
@Date: 24. 8. 14.
"""
import os
import pandas as pd

# CSV 파일 로드
mode = 'valid'

csv_file = f'{mode}_B.csv'
df = pd.read_csv(csv_file)
# 파일명만 추출
df['file_name'] = df['image_id'].apply(lambda x: os.path.basename(x))

# 폴더 경로 리스트
folders = [
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/흰색",
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/갈색",
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/검정",
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/연갈색",
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/흰색_검정",
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/검정_갈색",
    f"/home/divus/nas/yangsan/source/test/dogsss.v5i.folder/{mode}/흰색_갈색"
]

# 각 폴더에서 이미지 파일 리스트 가져오기
folder_images = {}
for folder in folders:
    folder_name = os.path.basename(folder)
    folder_images[folder_name] = []
    for file in os.listdir(folder):
        # '_jpg' 앞까지만 추출하여 저장
        base_file_name = file.split('_jpg')[0] + '.jpg'
        folder_images[folder_name].append(base_file_name)

# color 컬럼 초기화
df['color'] = None

# 매칭 작업 수행
for index, row in df.iterrows():
    for folder_name, images in folder_images.items():
        if row['file_name'] in images:
            df.at[index, 'color'] = folder_name
            break

# 결과 확인
df['color'] = df['color'].fillna(method='bfill')
print(df)

# 결과 저장
output_csv_file = f'{mode}_B.csv'
df.to_csv(output_csv_file, index=False)
