import os
import sys

import requests
import tqdm

model_checkpoints = [
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
]


def download_file(url: str, filename: str, with_progress: bool = True):
    with open(filename, "wb") as file_handle:
        with requests.get(url, stream=True) as req:
            total_size = int(req.headers.get("content-length", 0))
            block_size = 1024
            if not with_progress:
                file_handle.write(req.content)
            else:
                with tqdm.tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=filename.split("/")[-1],
                ) as progress_bar:
                    for data in req.iter_content(block_size):
                        progress_bar.update(len(data))
                        file_handle.write(data)

            if total_size != 0 and file_handle.tell() != total_size:
                print("ERROR, something went wrong")


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/sam"

    os.makedirs(model_dir, exist_ok=True)

    for url in model_checkpoints:
        filename = os.path.join(model_dir, url.split("/")[-1])
        download_file(url, filename)
