# you can find the list of pretrained model here: https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y
import os
import gdown


def g_down(url, output_path):
    gdown.download_folder(url, output=output_path, quiet=True,
                          use_cookies=False, remaining_ok=True)


def download_pretrained(version):
    if version == "v1":
        url = 'https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd?usp=sharing'
        output_path = '/localdata/asaadi/checkpoint_path/cp_hifigan'
        g_down(url, output_path)
        os.rename(os.path.join(output_path, "generator_v1"),
                  os.path.join(output_path, "g_gener_v1"))
    elif version == "v3":
        url = 'https://drive.google.com/drive/folders/1KKvuJTLp_gZXC8lug7H_lSXct38_3kx1?usp=share_link'
        output_path = '/localdata/asaadi/checkpoint_path_v3/cp_hifigan'
        g_down(url, output_path)
        os.rename(os.path.join(output_path, "generator_v3"),
                  os.path.join(output_path, "g_gener_v3"))

    return output_path
