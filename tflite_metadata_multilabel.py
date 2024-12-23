from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os

# 모델 정보 생성
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Multi-label image classifier"
model_meta.description = ("Classify images into categories based on fur length, breed, and weight. "
                          "The model outputs three different values: fur length, breed, and weight (regression).")
model_meta.version = "v1"
model_meta.author = "Your Name"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")

# 입력 텐서 정보 생성
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(224, 224))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean =[0.485]
input_normalization.options.std = [0.229]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [1.0]
input_stats.min = [0.0]
input_meta.stats = input_stats

# 털길이 출력 정보 생성
fur_length_output_meta = _metadata_fb.TensorMetadataT()
fur_length_output_meta.name = "fur_length"
fur_length_output_meta.description = "The length of the fur classified."
fur_length_output_meta.content = _metadata_fb.ContentT()
fur_length_output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
fur_length_output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
fur_length_output_stats = _metadata_fb.StatsT()
fur_length_output_stats.max = [1.0]
fur_length_output_stats.min = [0.0]
fur_length_output_meta.stats = fur_length_output_stats

# 라벨 파일 연결 (털길이)
fur_length_label_file = _metadata_fb.AssociatedFileT()
fur_length_label_file.name = os.path.basename("fur_length_labels.txt")
fur_length_label_file.description = "Labels for the fur lengths."
fur_length_label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
fur_length_output_meta.associatedFiles = [fur_length_label_file]

# 품종 출력 정보 생성
breed_output_meta = _metadata_fb.TensorMetadataT()
breed_output_meta.name = "breed"
breed_output_meta.description = "The breed of the animal classified."
breed_output_meta.content = _metadata_fb.ContentT()
breed_output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
breed_output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
breed_output_stats = _metadata_fb.StatsT()
breed_output_stats.max = [1.0]
breed_output_stats.min = [0.0]
breed_output_meta.stats = breed_output_stats

# 라벨 파일 연결 (품종)
breed_label_file = _metadata_fb.AssociatedFileT()
breed_label_file.name = os.path.basename("breed_labels.txt")
breed_label_file.description = "Labels for the breeds."
breed_label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
breed_output_meta.associatedFiles = [breed_label_file]

# 몸무게 출력 정보 생성
weight_output_meta = _metadata_fb.TensorMetadataT()
weight_output_meta.name = "weight"
weight_output_meta.description = "The weight of the animal in kilograms (regression output)."
weight_output_meta.content = _metadata_fb.ContentT()
weight_output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
weight_output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
weight_output_stats = _metadata_fb.StatsT()
weight_output_stats.max = [100.0]  # 실제 모델 범위에 따라 조정
weight_output_stats.min = [0.0]
weight_output_meta.stats = weight_output_stats

# 털색깔 출력 정보 생성
color_output_meta = _metadata_fb.TensorMetadataT()
color_output_meta.name = "color"
color_output_meta.description = "The color of classified."
color_output_meta.content = _metadata_fb.ContentT()
color_output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
color_output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
color_output_stats = _metadata_fb.StatsT()
color_output_stats.max = [1.0]
color_output_stats.min = [0.0]
color_output_meta.stats = color_output_stats

# 라벨 파일 연결 (털색깔)
color_label_file = _metadata_fb.AssociatedFileT()
color_label_file.name = os.path.basename("color_labels.txt")
color_label_file.description = "Labels for the color."
color_label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
color_output_meta.associatedFiles = [color_label_file]

# 서브그래프 정보 결합
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [breed_output_meta, fur_length_output_meta, weight_output_meta, color_output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

# 메타데이터 및 관련 파일 모델에 포함
model_file = "/home/divus/leeys/dogs/pet_id/saved_model/dogs14_float32.tflite"
# export_model_path = "/home/divus/leeys/dogs/pet_id/saved_model/dogs14_float32.tflite"
#
# populator = _metadata.MetadataPopulator.with_model_file(model_file)
# populator.load_metadata_buffer(metadata_buf)
# populator.load_associated_files(["fur_length_labels.txt", "breed_labels.txt", "color_labels.txt"])
# populator.populate()
#
# # 메타데이터 시각화
# displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
# export_json_file = os.path.join("export_directory",
#                                 os.path.splitext(os.path.basename(export_model_path))[0] + ".json")
# json_file = displayer.get_metadata_json()

# Optional: write out the metadata as a json file
# with open(export_json_file, "w") as f:
#     f.write(json_file)
