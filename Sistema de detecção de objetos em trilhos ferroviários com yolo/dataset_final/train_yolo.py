from ultralytics import YOLO  # type: ignore
import os
import torch  # type: ignore

if __name__ == "__main__":
    config = {
        'data': r'D:/projeto2/dataset_final/data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': '0',
        'name': 'rail_detect_yolov8s_v2',
        'patience': 10,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'amp': True,
        'workers': 8,
        'cache': False,
        'seed': 42,
        'single_cls': False,
        'augment': True
    }


    # Verificação do ambiente
    print("=== Verificação de Hardware ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Verificação do dataset
    print("\n=== Verificação do Dataset ===")
    print(f"Arquivo YAML: {os.path.exists(config['data'])}")
    print("Imagens treino:", len(os.listdir(r'D:/projeto2/dataset_final/images/train')))
    print("Imagens validação:", len(os.listdir(r'D:/projeto2/dataset_final/images/val')))

    # Treinamento
    try:
        model = YOLO('yolov8s.pt')
        results = model.train(
            **config,
        )

        # Exportação para TensorRT
        '''model.export(
            format='engine',
            device='0',
            simplify=True,
            workspace=4,
            half=True
        )'''

        print(f"\n✅ Treino completo! Modelos salvos em: runs/detect/{config['name']}")
        print(f"• Melhor modelo: runs/detect/{config['name']}/weights/best.engine")
        print(f"• Para inferência rápida: model = YOLO('runs/detect/{config['name']}/weights/best.engine')")

    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        if "CUDA out of memory" in str(e):
            print("\nSoluções:")
            print("1. Reduza batch_size para 4 ou 2")
            print("2. Diminua imgsz para 512")
            print("3. Desative amp: amp=False")
