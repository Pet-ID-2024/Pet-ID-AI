import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, mean_squared_error
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_breed = 0
        accuracy_hair = 0
        loss_weight = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            # target_labels['weight_labels'] = target_labels['weight_labels'].clone().detach().view(-1, 1)
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_breed, batch_accuracy_hair, batch_loss_weight = \
                calculate_metrics(output, target_labels)

            accuracy_breed += batch_accuracy_breed
            accuracy_hair += batch_accuracy_hair
            loss_weight += batch_loss_weight

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_breed /= n_samples
    accuracy_hair /= n_samples
    loss_weight /= n_samples

    print('-' * 72)
    print("Validation  loss: {:.4f}, breed: {:.4f}, hair: {:.4f}, weight: {:.4f}\n".format(
        avg_loss, accuracy_breed, accuracy_hair, loss_weight))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_breed', accuracy_breed, iteration)
    logger.add_scalar('val_accuracy_hair', accuracy_hair, iteration)
    logger.add_scalar('val_loss_weight', loss_weight, iteration)
    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_breed_all = []
    gt_hair_all = []
    gt_article_all = []
    predicted_breed_all = []
    predicted_hair_all = []
    predicted_article_all = []

    accuracy_breed = 0
    accuracy_hair = 0
    accuracy_article = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_breeds = batch['labels']['breed_labels']
            gt_hairs = batch['labels']['hair_labels']
            output = model(img.to(device))

            batch_accuracy_breed, batch_accuracy_hair, batch_accuracy_article = \
                calculate_metrics(output, batch['labels'])
            accuracy_breed += batch_accuracy_breed
            accuracy_hair += batch_accuracy_hair

            # get the most confident prediction for each image
            _, predicted_breeds = output['breed'].cpu().max(1)
            _, predicted_hairs = output['hair'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_breed = attributes.breed_id_to_name[predicted_breeds[i].item()]
                predicted_hair = attributes.hair_id_to_name[predicted_hairs[i].item()]

                gt_breed = attributes.breed_id_to_name[gt_breeds[i].item()]
                gt_hair = attributes.hair_id_to_name[gt_hairs[i].item()]

                gt_breed_all.append(gt_breed)
                gt_hair_all.append(gt_hair)

                predicted_breed_all.append(predicted_breed)
                predicted_hair_all.append(predicted_hair)

                imgs.append(image)
                labels.append("{}\n{}".format(predicted_hair, predicted_breed))
                gt_labels.append("{}\n{}".format(gt_hair, gt_breed))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nbreed: {:.4f}, hair: {:.4f}".format(
            accuracy_breed / n_samples,
            accuracy_hair / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # breed
        cn_matrix = confusion_matrix(
            y_true=gt_breed_all,
            y_pred=predicted_breed_all,
            labels=attributes.breed_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.breed_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("breeds")
        plt.tight_layout()
        plt.show()

        # hair
        cn_matrix = confusion_matrix(
            y_true=gt_hair_all,
            y_pred=predicted_hair_all,
            labels=attributes.hair_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.hair_labels).plot(
            xticks_rotation='horizontal')
        plt.title("hairs")
        plt.tight_layout()
        plt.show()

        # Uncomment code below to see the article confusion matrix (it may be too big to display)
        # cn_matrix = confusion_matrix(
        #     y_true=gt_article_all,
        #     y_pred=predicted_article_all,
        #     labels=attributes.article_labels,
        #     normalize='true')
        # plt.rcParams.update({'font.size': 1.8})
        # plt.rcParams.update({'figure.dpi': 300})
        # ConfusionMatrixDisplay(cn_matrix, attributes.article_labels).plot(
        #     include_values=False, xticks_rotation='vertical')
        # plt.rcParams.update({'figure.dpi': 100})
        # plt.rcParams.update({'font.size': 5})
        # plt.title("Article types")
        # plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()


def calculate_metrics(output, target):
    _, predicted_breed = output['breed'].cpu().max(1)
    gt_breed = target['breed_labels'].cpu()

    _, predicted_hair = output['hair'].cpu().max(1)
    gt_hair = target['hair_labels'].cpu()

    # _, predicted_weight = output['weight'].cpu().max(1)
    # gt_weight = target['weight_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_breed = balanced_accuracy_score(y_true=gt_breed.numpy(), y_pred=predicted_breed.numpy())
        accuracy_hair = balanced_accuracy_score(y_true=gt_hair.numpy(), y_pred=predicted_hair.numpy())
        # loss_weight = balanced_accuracy_score(y_true=gt_weight.numpy(), y_pred=predicted_weight.numpy())

    return accuracy_breed, accuracy_hair


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./fashion-product-images/styles.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(n_breed_classes=attributes.num_breeds, n_hair_classes=attributes.num_hairs).to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, device, checkpoint=args.checkpoint)
