import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, n_breed_classes, n_hair_classes):
        super().__init__()
        self.base_model = models.mobilenet_v3_small(pretrained=True, progress=True).features  # take the model without classifier
        last_channel = 576

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.breed = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=last_channel, out_features=n_breed_classes)
        )
        self.hair = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=last_channel, out_features=n_hair_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'breed': self.breed(x),
            'hair': self.hair(x)
        }

    def get_loss(self, net_output, ground_truth):
        breed_loss = F.cross_entropy(net_output['breed'], ground_truth['breed_labels'])
        hair_loss = F.cross_entropy(net_output['hair'], ground_truth['hair_labels'])
        loss = breed_loss + hair_loss
        return loss, {'breed': breed_loss, 'hair': hair_loss}
