import os
import random
import ast
import gc
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from Inpainting.model import CustomModel
from Inpainting.dataset.dataset import AudioDataset
from Inpainting.loss_fn import LossFunction
from Inpainting.dataset.km_label import ApplyKmeans
from Inpainting.utils import choose_device


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(mask_length=2):

    # print all configs :
    dataset_config = './dataset/config.yaml'
    with open(dataset_config, 'r') as file:
        data = yaml.safe_load(file)
        print("Dataset Configs:")
        print(data)

    training_config = 'config.yaml'
    with open(training_config, 'r') as file:
        data = yaml.safe_load(file)
        print("Training Configs:")
        print(data)

    predict_config = 'predict.yaml'
    with open(predict_config, 'r') as file:
        data = yaml.safe_load(file)
        print("Prediction Configs:")
        print(data)

    file_path = 'config.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    print(data)
    seed_all(data['training_config']['seed'])
    torch.cuda.empty_cache()
    gc.collect()
    dataset_name = data['training_config']['dataset']
    min_mask_length = data['training_config']['min_mask_length']//20
    max_mask_length = data['training_config']['max_mask_length']//20
    epochs = data['training_config']['epochs']
    max_wav_len = int(data['training_config']['max_wav_length']*16000)

    # Training/Validation Dataset:
    path2splits = os.path.abspath(
        data['training_dataset'][dataset_name]['path2splits'])
    path2pt = os.path.abspath(
        data['training_dataset'][dataset_name]['path2pt'])
    n_clusters = data['km_model']['n_clusters']
    path2centroids = os.path.join(data['training_dataset'][dataset_name]
                                  ['path2centroids'], f'km_model_{n_clusters}/label_dir/training')
    path2wavs = os.path.abspath(data['wave'][dataset_name]['path2wavs'])
    train_dataset = AudioDataset(path2splits=path2splits, path2wavs=path2wavs, path2pt=path2pt, path2centroids=path2centroids,
                                 min_mask_len=min_mask_length, max_mask_len=max_mask_length, epoch_num=0, epochs=epochs, max_wav_len=max_wav_len)

    path2splits = os.path.abspath(
        data['validation_dataset'][dataset_name]['path2splits'])
    path2pt = os.path.abspath(
        data['validation_dataset'][dataset_name]['path2pt'])
    path2centroids = os.path.join(data['validation_dataset'][dataset_name]
                                  ['path2centroids'], f'km_model_{n_clusters}/label_dir/validation')

    valid_dataset = AudioDataset(path2splits=path2splits, path2wavs=path2wavs, path2pt=path2pt, path2centroids=path2centroids,
                                 min_mask_len=min_mask_length, max_mask_len=max_mask_length, epoch_num=0, epochs=epochs)

    # DataLoader
    train_batch_size = data['training_config']['train_batch_size']
    valid_batch_size = data['training_config']['valid_batch_size']

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                  num_workers=4
                                  )
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, pin_memory=True,
                                  num_workers=4
                                  )
    num_epochs = data["training_config"]["epochs"]
    loss_function = data["training_config"]["loss_function"]
    # Model
    model = CustomModel(codebook_dim=data['model']['codebook_dim'], load_pretrained=data['model']['load_pretrained'],
                        type=data['model']['type'], train_encoder=data['model']['train_encoder'], loss_function=loss_function)

    total_params, trainable_params = count_parameters(model)
    print(f'Total number of parameters: {total_params}')
    print(f'Number of trainable parameters: {trainable_params}')

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': float(
            data['optimizer']['base_lr'])},
        {'params': model.final_layers.parameters(
        ), 'lr': float(data['optimizer']['fc_lr'])}
    ], betas=ast.literal_eval(data['optimizer']['betas']), eps=float(data['optimizer']['eps']), weight_decay=float(data['optimizer']['weight-decay']))
    max_norm = data['optimizer']['clip-norm']

    device = choose_device(data['training_config']['gpu_index'])
    print("Currently using : ", device)
    model.to(device)

    km_model_path = os.path.join(
        data['km_model'][dataset_name]['km_model_path'], f'km_model_{n_clusters}/model.km')
    loss_instance = LossFunction(km_model_path, device=device)

    # Train loop
    save_checkpoint = data['hubert_model']['save_checkpoint']
    save_training_checkpoint = data['hubert_model']['save_checkpoint_training']
    model_checkpoint = data['hubert_model']['model_checkpoint']
    if data['model']['load_pretrained']:
        if os.path.exists(model_checkpoint):
            print('Loading ' + model_checkpoint + '\n')
            model.load_state_dict(torch.load(
                model_checkpoint, map_location='cuda'))
        else:
            print('No saved model (from previous experiment) found, creating a new one\n')

    # model.load_state_dict(torch.load(model_checkpoint, map_location = 'cuda'))
    # best_valid_loss = float('inf')
    best_valid_acc = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        running_acc = 0.0
        running_cos_sim_acc = 0.0
        train_loss = 0.0
        train_acc = 0.0
        # model.train()
        train_dataset.epoch_num = epoch
        for batch_idx, batch_data in enumerate(train_dataloader):
            _, inputs, attention_masks, masks_pos, masks_len, labels = batch_data

            inputs = torch.autograd.Variable(
                inputs.to(device, non_blocking=True))
            attention_masks = torch.autograd.Variable(
                attention_masks.to(device, non_blocking=True))
            masks_pos = torch.autograd.Variable(
                masks_pos.to(device, non_blocking=True))
            masks_len = torch.autograd.Variable(
                masks_len.to(device, non_blocking=True))
            labels = torch.autograd.Variable(
                labels.to(device, non_blocking=True))

            optimizer.zero_grad()

            outputs = model(inputs, attention_masks)
            values = torch.zeros(
                (masks_pos.shape[0], masks_len[0], outputs.shape[-1])).to(device)
            for i in range(masks_pos.shape[0]):  # considre flatten the tensor
                values[i, :, :] = outputs[i, masks_pos[i]
                    :masks_pos[i]+masks_len[i], :]

            # Compute loss
            if loss_function == 'cos_sim':
                loss, pred_labels = loss_instance.cos_sim(values, labels)
            elif loss_function == 'mse':
                loss, pred_labels = loss_instance.MSELoss(values, labels)
            else:
                loss, pred_labels = loss_instance.soft_loss(values, labels)

            # Backward pass and optimization
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm)

            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            running_acc += torch.sum(pred_labels == labels)
            cos_sim_pred_target = loss_instance.cos_sim_target_labels(
                pred_labels, labels)
            running_cos_sim_acc += torch.sum(cos_sim_pred_target >= 0.95)
            train_acc += torch.sum(pred_labels == labels)
            if batch_idx % 100 == 99:
                print(f"Batch {batch_idx+1}, Loss: {running_loss/(train_batch_size*100*mask_length):.4f}\t",
                      f"Acc: {running_acc/(train_batch_size*100*mask_length):.4f}\t",
                      f"Acc Cos Sim: {running_cos_sim_acc/(train_batch_size*100*mask_length):.4f}"
                      )
                running_loss = 0.0
                running_acc = 0.0
                running_cos_sim_acc = 0.0

            if batch_idx % 100 == 99:
                model.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                valid_cos_sim_acc = 0.0
                with torch.no_grad():
                    for valid_batch_idx, valid_batch_data in enumerate(valid_dataloader):
                        # Forward pass
                        _, inputs, attention_masks, masks_pos, masks_len, labels = valid_batch_data
                        inputs = inputs.to(device)
                        attention_masks = attention_masks.to(device)
                        masks_pos = masks_pos.to(device)
                        masks_len = masks_len.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs, attention_masks)
                        values = torch.zeros(
                            (masks_pos.shape[0], masks_len[0], outputs.shape[-1])).to(device)
                        for i in range(masks_pos.shape[0]):
                            values[i, :, :] = outputs[i, masks_pos[i]:masks_pos[i]+masks_len[i], :]

                        # Compute loss
                        if loss_function == 'cos_sim':
                            loss, pred_labels = loss_instance.cos_sim(
                                values, labels)
                        elif loss_function == 'mse':
                            loss, pred_labels = loss_instance.MSELoss(
                                values, labels)
                        else:
                            loss, pred_labels = loss_instance.soft_loss(
                                values, labels)

                        # Update the validation loss
                        valid_loss += loss.item()
                        valid_acc += torch.sum(pred_labels == labels)
                        cos_sim_pred_target = loss_instance.cos_sim_target_labels(
                            pred_labels, labels)
                        valid_cos_sim_acc += torch.sum(
                            cos_sim_pred_target >= 0.95)

                    # Compute the average validation loss
                    valid_loss /= len(valid_dataloader.dataset)*mask_length
                    valid_acc /= len(valid_dataloader.dataset)*mask_length
                    valid_cos_sim_acc /= len(valid_dataloader.dataset) * \
                        mask_length
                    print(f"Validation loss: {valid_loss:.4f}\t",
                          f"Validation Acc: {valid_acc:.4f}\t",
                          f"Validation Acc Cos Sim: {valid_cos_sim_acc:.4f}"
                          )

                    if valid_cos_sim_acc > best_valid_acc:
                        best_valid_acc = valid_cos_sim_acc
                        print("Saving model...\t")
                        if not os.path.exists(os.path.dirname(save_checkpoint)):
                            os.makedirs(os.path.dirname(save_checkpoint))
                        torch.save(model.state_dict(), save_checkpoint)
                        print("Done")
            model.train()
        train_loss /= len(train_dataloader.dataset)*mask_length
        train_acc /= len(train_dataloader.dataset)*mask_length
        print(f"Training loss for the full epoch: {train_loss:.4f}\t",
              f"Training acc for the full epoch: {train_acc:.4f}")

    print("Training finished!")
    torch.save(model.state_dict(), save_training_checkpoint)


def save_fig(mel_spec, fig_name):
    fig, ax = plt.subplots()
    im = ax.imshow(mel_spec.numpy(), cmap='inferno')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title('Mel Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency')
    ax.set_xlim([0, mel_spec.shape[1]-1])
    cbar.ax.set_ylabel('Intensity')
    fig.savefig('images/'+fig_name + '.png')


def compute_acc(pred_mel, target_mel, km_path):
    km_model = ApplyKmeans(km_path)
    pred_mel = pred_mel.view(-1, pred_mel.shape[-1])
    target_mel = target_mel.view(-1, target_mel.shape[-1])

    pred_labels = torch.from_numpy(km_model(pred_mel))
    target_labels = torch.from_numpy(km_model(target_mel))

    correct = torch.sum(pred_labels == target_labels).item()
    total = target_labels.numel()
    accuracy = correct / total

    return accuracy


if __name__ == "__main__":
    mask_length = 20
    train(mask_length=mask_length)
