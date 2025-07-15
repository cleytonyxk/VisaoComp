import os

# Caminho onde estão as labels do seu dataset
LABELS_DIR = r"D:\projeto2\dataset_final\labels"

# Quantidade de classes COCO
OFFSET = 80

# Percorre todos os arquivos txt
for subset in ["train", "val"]:
    subset_dir = os.path.join(LABELS_DIR, subset)
    if not os.path.exists(subset_dir):
        print(f"⚠️ Pasta não encontrada: {subset_dir}. Pulando.")
        continue

    for file in os.listdir(subset_dir):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(subset_dir, file)
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"⚠️ Linha inválida em {file}: {line}")
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                print(f"⚠️ Classe inválida em {file}: {parts[0]}")
                continue

            # Soma o offset
            new_class_id = class_id + OFFSET
            # Recria a linha
            new_line = " ".join([str(new_class_id)] + parts[1:])
            new_lines.append(new_line)

        # Salva sobrescrevendo
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f"✅ Atualizado: {file_path}")

print("\n🎉 Reindexação concluída. Todos os labels do seu dataset agora começam no índice 80.")
