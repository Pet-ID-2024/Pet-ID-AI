import argparse
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from model_mobilev4 import MultiOutputModel
from test import calculate_metrics, validate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchPCGrad.pcgrad import PCGrad
import torch.optim.lr_scheduler as lr_scheduler


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
    N_epochs = 100
    batch_size = 256
    num_workers = 4  # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean, std)
    ])
    train_dataset = FashionDataset('./train_B.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = FashionDataset('./valid_B.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = MultiOutputModel(n_breed_classes=attributes.num_breed,
                             n_hair_classes=attributes.num_hair,
                             n_weight_classes=attributes.num_weight,
                             n_color_classes=attributes.num_color)\
                            .to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr = 0.0001))
    
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

    for epoch in tqdm(range(start_epoch, N_epochs + 1), desc='Training', leave=False):
        model.train()
        total_loss = 0
        accuracy_breed = 0
        accuracy_hair = 0
        loss_weight = 0
        accuracy_color = 0

        # tqdm for the batch loop
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{N_epochs}', unit='batch', leave=False) as pbar:
            for batch in train_dataloader:
                optimizer.zero_grad()

                img = batch['img'].to(device)

                target_labels = {t: batch['labels'][t].to(device) for t in batch['labels']}
                target_labels['weight_labels'] = target_labels['weight_labels'].clone().detach().view(-1, 1).to(device).float()


                # print(target_labels)
                output = model(img)

                loss_train, losses_train, loss_list = model.get_loss(output, target_labels)
                total_loss += loss_train.item()

                weight_loss = losses_train['weight'].cpu().detach()

                batch_accuracy_breed, batch_accuracy_hair, batch_loss_weight, batch_accuracy_color = calculate_metrics(output, target_labels)
                accuracy_breed += batch_accuracy_breed
                accuracy_hair += batch_accuracy_hair
                loss_weight += weight_loss
                accuracy_color += batch_accuracy_color
                # print(batch_loss_weight)
                # print(loss_weight)
                
                # optimizer.pc_backward(loss_list)
                loss_train.backward()
                optimizer.step()

                # Update tqdm progress bar
                pbar.update(1)

        # Logging the average metrics per epoch
        avg_loss = total_loss / n_train_samples
        avg_accuracy_breed = accuracy_breed / n_train_samples
        avg_accuracy_hair = accuracy_hair / n_train_samples
        avg_loss_weight = loss_weight / n_train_samples
        avg_accuracy_color = accuracy_color / n_train_samples

        print(
            f"Epoch {epoch:4d}, Loss: {avg_loss:.4f}, Breed: {avg_accuracy_breed:.4f}, Hair: {avg_accuracy_hair:.4f}, Weight: {avg_loss_weight:.4f}, Color: {avg_accuracy_color:.4f}")

        logger.add_scalar('train_loss', avg_loss, epoch)

        # Validation and checkpoint saving every 5 epochs
        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)
            checkpoint_save(model, savedir, epoch)
