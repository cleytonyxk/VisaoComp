import os
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

# 🟨 CAMINHOS
COCO_DIR = r"D:\projeto2\coco_yolo\coco"  # caminho para o diretório extraído
OUTPUT_DIR = r"D:\projeto2\coco_yolo\coco_yolo_format"

# Cria as pastas de saída
for subset in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', subset), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', subset), exist_ok=True)

# ⚙️ Função para conversão
def convert_subset(subset):
    ann_file = os.path.join(COCO_DIR, 'annotations', f'instances_{subset}2017.json')
    coco = COCO(ann_file)

    img_dir = os.path.join(COCO_DIR, f'{subset}2017')
    out_img_dir = os.path.join(OUTPUT_DIR, 'images', subset)
    out_label_dir = os.path.join(OUTPUT_DIR, 'labels', subset)

    categories = coco.loadCats(coco.getCatIds())
    category_map = {cat['id']: idx for idx, cat in enumerate(categories)}  # id original → índice YOLO

    for img_id in tqdm(coco.getImgIds(), desc=f"Convertendo {subset}"):
        img = coco.loadImgs(img_id)[0]
        file_name = img['file_name']
        width = img['width']
        height = img['height']

        # Copia imagem
        src_img_path = os.path.join(img_dir, file_name)
        dst_img_path = os.path.join(out_img_dir, file_name)
        shutil.copy2(src_img_path, dst_img_path)

        # Cria o arquivo de label
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        label_file = os.path.join(out_label_dir, file_name.replace('.jpg', '.txt'))

        with open(label_file, 'w') as f:
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height
                class_id = category_map[ann['category_id']]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    return [cat['name'] for cat in categories]

# 🧠 Executa
if __name__ == "__main__":
    print("🔁 Iniciando conversão do COCO para YOLO...")
    class_names = convert_subset("train")
    convert_subset("val")

    # Cria o arquivo data.yaml
    yaml_data = {
        'train': os.path.join(OUTPUT_DIR, 'images', 'train').replace("\\", "/"),
        'val': os.path.join(OUTPUT_DIR, 'images', 'val').replace("\\", "/"),
        'nc': len(class_names),
        'names': class_names
    }

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        import yaml
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ Conversão concluída. Arquivo data.yaml salvo em {OUTPUT_DIR}")
