import os
import matplotlib.pyplot as plt
import torch


def choose_device(idx):
    if idx == 'cpu':
        device = torch.device("cpu")
        print(f"Using CPU: ", device)
        return device
    if torch.cuda.is_available():
        # Get the total number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # List the available GPUs
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        # Choose a specific GPU
        if num_gpus < idx+1:
            return torch.device(f"cuda:{num_gpus-1}")
        device = torch.device(f"cuda:{idx}")  # Specify the device
        print(f"Using GPU {idx}: {torch.cuda.get_device_name(idx)}")
        return device
    else:
        print("No GPUs available. Using CPU.")
        device = torch.device("cpu")
        return device


def cos_sim(tensor1, tensor2, x1=None, x2=None, figure_name='Test'):
    tensor1 = tensor1.squeeze(0).transpose(0, 1)
    tensor2 = tensor2.squeeze(0).transpose(0, 1)

    # Calculate cosine similarity between each pair of rows
    cos_similarities = torch.nn.functional.cosine_similarity(
        tensor1, tensor2, dim=1)
    # Plot the cosine similarities
    fig = plt.figure()
    plt.plot(cos_similarities, label=figure_name, color='b')
    if x1 is not None and x2 is not None:
        plt.axvline(x=x1, color='r', linestyle='--')
        plt.axvline(x=x2, color='r', linestyle='--')
        plt.xticks([0, x1, x2, tensor1.shape[0]], [
                   '0', 's', 'e', str(tensor1.shape[0])])
        plt.xlim([x1-5, x2+5])
        if x1 > 0:
            print("Cos sim for s-1: ", cos_similarities[x1-1])
        print("Cos sim for s: ", cos_similarities[x1])
        print("Cos sim for e: ", cos_similarities[x2])
        if x2 < cos_similarities.shape[0]-1:
            print("Cos sim for e+1: ", cos_similarities[x2+1])

    else:
        plt.xticks([0, tensor1.shape[0]], ['0', str(tensor1.shape[0])])
    plt.xlabel('Time (*20 ms)')
    plt.ylabel('Cosine Similarity')
    plt.show()
    plt.ylim([-1.1, 1.1])
    # plt.legend(figure_name)
    plt.title(figure_name)
    plt.savefig(os.path.join(
        '/nethome/asaadi/speechinpaint/Speech_Inpainting/ihab_hubert/dataset/images', figure_name+'.png'))
    print("Saved successfully")


def mask_audio(audio, mask_pos, mask_length):
    audio[mask_pos:mask_pos+mask_length] = 0
    return audio
