import os
import requests
from tqdm import tqdm
import zipfile

# Links oficiais COCO
COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Diretório base
BASE_DIR = r"D:\projeto2\coco_yolo"
COCO_DIR = os.path.join(BASE_DIR, "coco")
os.makedirs(COCO_DIR, exist_ok=True)

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Baixando {os.path.basename(dest_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Extraindo {os.path.basename(zip_path)}...")
        zip_ref.extractall(extract_to)

def main():
    for filename, url in COCO_URLS.items():
        dest_zip = os.path.join(COCO_DIR, filename)
        if not os.path.exists(dest_zip):
            download_file(url, dest_zip)
        else:
            print(f"{filename} já existe, pulando download.")
        extract_zip(dest_zip, COCO_DIR)
        print(f"{filename} concluído.\n")

    print("✅ Todos os arquivos baixados e extraídos.")

if __name__ == "__main__":
    main()


