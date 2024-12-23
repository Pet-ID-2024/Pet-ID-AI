# import pandas as pd
# import json
# import natsort
# import glob
# import os
#
# image_path = '/media/leeyongseong/새 볼륨/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_B_반려견'
# json_data = "/media/leeyongseong/새 볼륨/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TS_B_반려견"
# json_list = natsort.natsorted(glob.glob(os.path.join(json_data, '*.json')))
#
# breed_list = []
# hair_list = []
# # weight_list = []
# image_list = []
#
# extract_list = ['train_B', 'valid_B']
#
# for ex in extract_list:
#     breed_list = []
#     hair_list = []
#     classes_list = []
#     weight_list = []
#     image_list = []
#
#     if ex == 'train_B':
#         image_path = f'/media/leeyongseong/새 볼륨/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_B_반려견'
#         json_data = f"/media/leeyongseong/새 볼륨/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_B_반려견"
#     elif ex == 'valid_B':
#         image_path = f'/media/leeyongseong/새 볼륨/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_B_반려견'
#         json_data = f"/media/leeyongseong/새 볼륨/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_B_반려견"
#
#     json_list = natsort.natsorted(glob.glob(os.path.join(json_data, '*.json')))
#     for js in json_list:
#         with open(js, 'r', encoding='utf-8') as file:
#             data = file.read()
#
#         data = json.loads(data)
#
#         breed = data['metadata']['id']['breed']
#         hair = data['metadata']['id']['group']
#         classes = data['metadata']['id']['class']
#         weight = data['metadata']['physical']['weight']
#         image_id = data['annotations']['image-id']
#         image_id = os.path.join(image_path, image_id)
#
#         if not os.path.exists(image_id):
#             print(f"File does not exist: {image_id}")
#             continue
#
#         breed_list.append(breed)
#         hair_list.append(hair)
#         weight_list.append(weight)
#         classes_list.append(classes)
#         image_list.append(image_id)
#
#     result = pd.DataFrame({
#         'image_id' : image_list,
#         'breed' : breed_list,
#         'hair' : hair_list,
#         'classes' : classes_list,
#         'weight' : weight_list
#         })
#
#     result.to_csv(f'{ex}.csv')
#
#
#

# import pandas as pd
# import json
# import natsort
# import glob
# import os

# extract_list = ['train_B', 'valid_B']

# for ex in extract_list:
#     breed_list = []
#     hair_list = []
#     classes_list = []
#     weight_list = []
#     image_list = []

#     if ex == 'train_B':
#         image_folders = [
#             '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_A_반려견_1',
#             '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_A_반려견_2',
#             '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_A_반려견_3',
#             '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_B_반려견'
#         ]
#         json_data_folders = [
#             "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_B_반려견",
#             "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_A_반려견"
#         ]
#     elif ex == 'valid_B':
#         image_folders = [
#             '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_B_반려견',
#             '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_A_반려견'
#         ]
#         json_data_folders = [
#             "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_B_반려견",
#             "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_A_반려견"
#         ]

#     json_list = []
#     for json_folder in json_data_folders:
#         json_list.extend(natsort.natsorted(glob.glob(os.path.join(json_folder, '*.json'))))

#     for js in json_list:
#         with open(js, 'r', encoding='utf-8') as file:
#             data = file.read()

#         data = json.loads(data)

#         breed = data['metadata']['id']['breed']
#         hair = data['metadata']['id']['group']
#         classes = data['metadata']['id']['class']
#         weight = data['metadata']['physical']['weight']
#         image_id = data['annotations']['image-id']

#         img_num = image_id.split('.')[0].split('_')[-1]
#         if int(img_num) >= 2:
#             continue

#         image_found = False
#         for image_path in image_folders:
#             full_image_path = os.path.join(image_path, image_id)
#             if os.path.exists(full_image_path):
#                 image_found = True
#                 break

#         if not image_found:
#             print(f"File does not exist: {image_id} in any of the specified folders")
#             continue

#         breed_list.append(breed)
#         hair_list.append(hair)
#         weight_list.append(weight)
#         classes_list.append(classes)
#         image_list.append(full_image_path)

#     result = pd.DataFrame({
#         'image_id': image_list,
#         'breed': breed_list,
#         'hair': hair_list,
#         'classes': classes_list,
#         'weight': weight_list
#     })

#     result.to_csv(f'{ex}.csv', index=False)

import pandas as pd
import json
import natsort
import glob
import os
from PIL import Image, ExifTags

def correct_image_rotation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None:
            orientation = exif.get(orientation, None)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image

extract_list = ['train_B', 'valid_B']

for ex in extract_list:
    breed_list = []
    hair_list = []
    classes_list = []
    weight_list = []
    image_list = []
    cropped_image_list = []  # To store paths of cropped images

    if ex == 'train_B':
        image_folders = [
            '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_A_반려견_1',
            '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_A_반려견_2',
            '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_A_반려견_3',
            '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_B_반려견'
        ]
        json_data_folders = [
            "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_B_반려견",
            "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_A_반려견"
        ]
    elif ex == 'valid_B':
        image_folders = [
            '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_B_반려견',
            '/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_A_반려견'
        ]
        json_data_folders = [
            "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_B_반려견",
            "/home/divus/nas/yangsan/source/test/204.반려견, 반려묘 건강정보 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_A_반려견"
        ]

    json_list = []
    for json_folder in json_data_folders:
        json_list.extend(natsort.natsorted(glob.glob(os.path.join(json_folder, '*.json'))))

    for js in json_list:
        with open(js, 'r', encoding='utf-8') as file:
            data = file.read()

        data = json.loads(data)

        breed = data['metadata']['id']['breed']
        print(breed)
        hair = data['metadata']['id']['group']
        classes = data['metadata']['id']['class']
        weight = data['metadata']['physical']['weight']
        image_id = data['annotations']['image-id']
        points = data['annotations']['label']['points']

        img_num = image_id.split('.')[0].split('_')[-1]
        if int(img_num) not in [1]:
            continue

        if breed == 'ETC' or breed == 'DRI':
            print('ETC 넘어감')
            continue

        if breed == 'DAL' or breed == 'DAS':
            print('DAL DAS 합침')
            breed = 'DAC'
        
        if breed == 'MIL' or breed == 'MIS':
            print('MIL MIS 합침')
            breed = 'MIX'

        if breed == 'CHS' or breed == 'CHL':
            print('CHS CHL 합침')
            breed = 'CHIW'

        image_found = False
        for image_path in image_folders:
            full_image_path = os.path.join(image_path, image_id)
            if os.path.exists(full_image_path):
                image_found = True
                break

        if not image_found:
            print(f"File does not exist: {image_id} in any of the specified folders")
            continue

        # Crop the image based on the points
        img = Image.open(full_image_path)
        img = correct_image_rotation(img) 
        x1, y1 = points[0]
        x2, y2 = points[1]
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Save the cropped image in the same directory as the original image
        cropped_image_path = os.path.join(os.path.dirname(full_image_path), f"cropped_{image_id}")
        cropped_img.save(cropped_image_path)

        breed_list.append(breed)
        hair_list.append(hair)
        weight_list.append(weight)
        classes_list.append(classes)
        image_list.append(full_image_path)
        cropped_image_list.append(cropped_image_path)

    result = pd.DataFrame({
        'image_id': image_list,
        'cropped_image_id': cropped_image_list,
        'breed': breed_list,
        'hair': hair_list,
        'classes': classes_list,
        'weight': weight_list
    })

    result.to_csv(f'{ex}.csv', index=False)
