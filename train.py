import argparse
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from test import calculate_metrics, validate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='./train_B.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 50
    batch_size = 16
    num_workers = 8  # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.breedJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Resize((640,640), antialias=True),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640), antialias=True),
        transforms.Normalize(mean, std)
    ])

    train_dataset = FashionDataset('./train_B.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = FashionDataset('./valid_B.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiOutputModel(n_breed_classes=attributes.num_breed,
                             n_hair_classes=attributes.num_hair)\
                            .to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    # print("\nAll hair labels:\n", attributes.hair_labels)
    # print("\nAll breed labels:\n", attributes.breed_labels)
    # print("\nAll article labels:\n", attributes.article_labels)

    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_breed = 0
        accuracy_hair = 0
        loss_weight= 0
        #tqdm(dataloader, desc="Training", leave=False)
        for batch in tqdm(train_dataloader, desc='Training', leave=False):
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            # target_labels['weight_labels'] = target_labels['weight_labels'].clone().detach().view(-1, 1)
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_breed, batch_accuracy_hair = \
                calculate_metrics(output, target_labels)

            accuracy_breed += batch_accuracy_breed
            accuracy_hair += batch_accuracy_hair
            # loss_weight += batch_mse_weight

            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, breed: {:.4f}, hair: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_breed / n_train_samples,
            accuracy_hair / n_train_samples
            ))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)
