import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Large_Weights

from two_way_loss import multilabel_categorical_crossentropy
from focal_loss import FocalLoss


class MultiOutputModel(nn.Module):
    def __init__(self, n_breed_classes, n_hair_classes, n_weight_classes, n_color_classes):
        super().__init__()
        # self.base_model = models.mobilenet_v3_small(pretrained=True, progress=True).features  # take the model without classifier
        # last_channel = 576

        efficientnet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.base_model = efficientnet.features  # take the model without classifier
        last_channel = efficientnet.classifier[0].in_features

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputsco
        self.breed = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=last_channel, out_features=n_breed_classes),
            # nn.Softmax(dim=1)
        )
        self.hair = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=last_channel, out_features=n_hair_classes),
            # nn.Softmax(dim=1)
        )
        self.weight = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=last_channel, out_features=1)
            # nn.Softmax(dim=1)
        )
        self.color = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=last_channel, out_features=n_color_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'breed': self.breed(x),
            'hair': self.hair(x),
            'weight': self.weight(x),
            'color': self.color(x)
        }

    # def get_loss(self, net_output, ground_truth):
    #     loss_fun = nn.MSELoss()

    #     breed_loss = F.cross_entropy(net_output['breed'], ground_truth['breed_labels'])
    #     hair_loss = F.cross_entropy(net_output['hair'], ground_truth['hair_labels'])
    #     weight_loss = loss_fun(net_output['weight'], ground_truth['weight_labels'])
    #     color_loss = F.cross_entropy(net_output['color'], ground_truth['color_labels'])

    #     loss = breed_loss + hair_loss + weight_loss + color_loss

    #     # alpha = (breed_loss + hair_loss) / loss
    #     # beta = weight_loss / loss

    #     # # 총 손실 계산
    #     # loss = alpha * (breed_loss + hair_loss) + beta * weight_loss
    #     # breed_loss = alpha * breed_loss
    #     # hair_loss = alpha * hair_loss
    #     # weight_loss = beta * weight_loss

    #     # print(loss)
    #     return loss.cpu(), {'breed': breed_loss, 'hair': hair_loss, 'weight' : weight_loss, 'color' : color_loss}, [breed_loss, hair_loss, weight_loss, color_loss]

    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def get_loss(self, net_output, ground_truth): # focal loss
        loss_fun = nn.MSELoss()
        focalloss_breed = FocalLoss()
        focalloss_hair = FocalLoss()

        breed_loss = F.cross_entropy(net_output['breed'], ground_truth['breed_labels'])
        hair_loss = F.cross_entropy(net_output['hair'], ground_truth['hair_labels'])
        weight_loss = loss_fun(net_output['weight'], ground_truth['weight_labels'])
        color_loss = F.cross_entropy(net_output['color'], ground_truth['color_labels'])

        loss = breed_loss + hair_loss + weight_loss + color_loss

        # alpha = (breed_loss + hair_loss) / loss
        # beta = weight_loss / loss

        # # 총 손실 계산
        # loss = alpha * (breed_loss + hair_loss) + beta * weight_loss
        # breed_loss = alpha * breed_loss
        # hair_loss = alpha * hair_loss
        # weight_loss = beta * weight_loss

        # print(loss)
        return loss.cpu(), {'breed': breed_loss, 'hair': hair_loss, 'weight' : weight_loss, 'color' : color_loss}, [breed_loss, hair_loss, weight_loss, color_loss]
