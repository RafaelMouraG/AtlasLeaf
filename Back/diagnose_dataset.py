"""
Diagnóstico de por que o modelo não reconhece bem fotos de campo.
"""
from PIL import Image
from pathlib import Path

# Abre imagens do dataset de ferrugem
rust_dir = Path('datasets/unified/asian_rust')
imgs = list(rust_dir.glob('asdid*.jpg'))[:3]

print('=== Imagens do dataset de ferrugem (ASDID) ===')
for img_path in imgs:
    img = Image.open(img_path)
    # Cores médias
    r, g, b = img.split()
    r_vals = list(r.getdata())
    g_vals = list(g.getdata())
    b_vals = list(b.getdata())
    r_mean = sum(r_vals) / len(r_vals)
    g_mean = sum(g_vals) / len(g_vals)
    b_mean = sum(b_vals) / len(b_vals)
    print(f'{img_path.name}:')
    print(f'  Tamanho: {img.size}')
    print(f'  RGB médio: ({r_mean:.0f}, {g_mean:.0f}, {b_mean:.0f})')
    # Verde dominante?
    if g_mean > r_mean and g_mean > b_mean:
        print(f'  Cor dominante: VERDE (folha)')
    elif b_mean > r_mean and b_mean > g_mean:
        print(f'  Cor dominante: AZUL (céu?)')
    else:
        print(f'  Cor dominante: MARROM/AMARELO (doença/senescência)')
    print()

# Compara com imagens do Kaggle (se existir algo diferente)
kaggle_imgs = list(rust_dir.glob('kaggle*.jpg'))[:3]
if kaggle_imgs:
    print('=== Imagens do Kaggle ===')
    for img_path in kaggle_imgs:
        img = Image.open(img_path)
        r, g, b = img.split()
        r_vals = list(r.getdata())
        g_vals = list(g.getdata())
        b_vals = list(b.getdata())
        r_mean = sum(r_vals) / len(r_vals)
        g_mean = sum(g_vals) / len(g_vals)
        b_mean = sum(b_vals) / len(b_vals)
        print(f'{img_path.name}:')
        print(f'  Tamanho: {img.size}')
        print(f'  RGB médio: ({r_mean:.0f}, {g_mean:.0f}, {b_mean:.0f})')
        print()
