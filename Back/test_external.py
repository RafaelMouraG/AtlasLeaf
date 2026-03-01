"""
Script para testar imagens externas de ferrugem.
Salve as imagens de teste em Back/test_images/ e execute este script.
"""
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from data_pipeline.model_v31 import create_model
from pathlib import Path
import onnxruntime as ort
import requests
from io import BytesIO

# Carrega modelo PyTorch
print("Carregando modelo...")
checkpoint = torch.load('atlasleaf_v31_best_model.pth', map_location='cpu', weights_only=False)
model = create_model(
    model_name=checkpoint['config']['model_name'],
    num_classes=checkpoint['config']['num_classes'],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open('atlasleaf_v31_metadata.json') as f:
    meta = json.load(f)
class_names = [c['name'] for c in meta['classes']]
friendly_names = [c['friendly_name'] for c in meta['classes']]

def test_image(img, name="test"):
    """Testa uma imagem e mostra Top-5"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    print(f"\n=== {name} ===")
    print(f"Tamanho original: {img.size}")
    
    top5 = probs.argsort(descending=True)[:5]
    for i, idx in enumerate(top5):
        marker = "👈" if class_names[idx] == "asian_rust" else ""
        print(f"  {i+1}. {friendly_names[idx]:30} {probs[idx].item():.2%} {marker}")
    
    return class_names[probs.argmax().item()]

# Testa com imagens do dataset para referência
print("\n" + "="*60)
print("REFERÊNCIA: Imagens do dataset de ferrugem")
print("="*60)

rust_dir = Path('datasets/unified/asian_rust')
for img_path in list(rust_dir.glob('*.jpg'))[:3]:
    img = Image.open(img_path)
    test_image(img, f"Dataset: {img_path.name}")

# Testa com imagens na pasta test_images/
print("\n" + "="*60)
print("TESTE: Imagens externas")
print("="*60)

test_dir = Path('test_images')
if test_dir.exists():
    test_images = list(test_dir.glob('*.[jp][pn][g]*')) + list(test_dir.glob('*.jpeg'))
    if test_images:
        for img_path in test_images:
            try:
                img = Image.open(img_path)
                test_image(img, f"Teste: {img_path.name}")
            except Exception as e:
                print(f"Erro ao carregar {img_path.name}: {e}")
    else:
        print("Nenhuma imagem encontrada em test_images/")
else:
    print("Pasta test_images/ não existe. Crie e coloque as imagens lá.")
    test_dir.mkdir(exist_ok=True)

# Compara com imagens de outras classes para ver se há confusão
print("\n" + "="*60)
print("COMPARAÇÃO: Outras doenças similares")
print("="*60)

# Cercospora (manchas marrons)
cercospora_dir = Path('datasets/unified/cercospora_blight')
if cercospora_dir.exists():
    imgs = list(cercospora_dir.glob('*.jpg'))[:2]
    for img_path in imgs:
        img = Image.open(img_path)
        test_image(img, f"Cercospora: {img_path.name}")

# Frogeye (olho de rã)
frogeye_dir = Path('datasets/unified/frogeye_leaf_spot')
if frogeye_dir.exists():
    imgs = list(frogeye_dir.glob('*.jpg'))[:2]
    for img_path in imgs:
        img = Image.open(img_path)
        test_image(img, f"Frogeye: {img_path.name}")
