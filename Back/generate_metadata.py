"""Script para gerar metadata v31"""
import json
from pathlib import Path
import torch

base_dir = Path(__file__).parent

# Carrega manifest existente
with open(base_dir / 'datasets/unified/manifest.json') as f:
    manifest = json.load(f)

# Carrega checkpoint para ver config
checkpoint = torch.load(base_dir / 'atlasleaf_v31_best_model.pth', map_location='cpu', weights_only=False)
train_config = checkpoint['config']

# Monta classes ordenadas por ID
class_dist = manifest['class_distribution']
classes = []

# Mapeamento de severidade conhecido
severity_map = {
    'healthy': 'none',
    'asian_rust': 'critical',
    'target_spot': 'high',
    'cercospora_blight': 'high',
    'septoria_brown_spot': 'medium',
    'bacterial_blight': 'medium',
    'powdery_mildew': 'low',
    'downy_mildew': 'medium',
    'soybean_mosaic': 'low',
    'yellow_mosaic': 'low',
    'sudden_death_syndrome': 'critical',
    'southern_blight': 'high',
    'potassium_deficiency': 'medium',
    'frogeye_leaf_spot': 'medium',
    'phytotoxicity': 'medium',
}

# Nome científico
scientific_names = {
    'healthy': 'N/A',
    'asian_rust': 'Phakopsora pachyrhizi',
    'target_spot': 'Corynespora cassiicola',
    'cercospora_blight': 'Cercospora kikuchii',
    'septoria_brown_spot': 'Septoria glycines',
    'bacterial_blight': 'Pseudomonas savastanoi pv. glycinea',
    'powdery_mildew': 'Microsphaera diffusa',
    'downy_mildew': 'Peronospora manshurica',
    'soybean_mosaic': 'Soybean mosaic virus (SMV)',
    'yellow_mosaic': 'Bean yellow mosaic virus (BYMV)',
    'sudden_death_syndrome': 'Fusarium virguliforme',
    'southern_blight': 'Sclerotium rolfsii',
    'potassium_deficiency': 'N/A',
    'frogeye_leaf_spot': 'Cercospora sojina',
    'phytotoxicity': 'N/A (dano químico)',
}

# Pega do manifest
for name, info in class_dist.items():
    classes.append({
        'id': str(info['id']),
        'name': name,
        'friendly_name': info['friendly_name'],
        'scientific_name': scientific_names.get(name, 'N/A'),
        'severity': severity_map.get(name, 'medium')
    })

# Ordena por ID
classes.sort(key=lambda x: int(x['id']))

# Verifica se phytotoxicity existe no disco mas não no manifest
unified_dir = base_dir / 'datasets/unified'
for folder in sorted(unified_dir.iterdir()):
    if folder.is_dir() and folder.name not in class_dist:
        next_id = len(classes)
        classes.append({
            'id': str(next_id),
            'name': folder.name,
            'friendly_name': 'Fitotoxicidade' if folder.name == 'phytotoxicity' else folder.name.replace('_', ' ').title(),
            'scientific_name': scientific_names.get(folder.name, 'N/A'),
            'severity': severity_map.get(folder.name, 'medium')
        })

# Monta metadata
metadata = {
    'project': 'AtlasLeaf',
    'version': '3.1',
    'task': 'multi-class disease classification',
    'model': 'efficientnet_b3',
    'dataset': 'unified',
    'total_images': manifest['total_images'],
    'classes': classes,
    'num_classes': train_config['num_classes'],
    'preprocessing': {
        'resize': train_config['input_size'],
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'input_shape': [1, 3, train_config['input_size'], train_config['input_size']],
    'metrics': {
        'val_accuracy': checkpoint['val_acc'],
        'val_loss': checkpoint['val_loss']
    },
    'training': {
        'epochs_trained': checkpoint['epoch'] + 1,
        'best_val_loss': checkpoint['val_loss'],
        'best_val_acc': checkpoint['val_acc'],
        'focal_gamma': train_config['focal_gamma_end'],
        'mixup_alpha': train_config['mixup_alpha'],
        'model_name': train_config['model_name'],
        'dropout': train_config['dropout'],
    }
}

# Salva
output_path = base_dir / 'atlasleaf_v31_metadata.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f'✅ {output_path.name} criado!')
print(f'   Classes: {len(classes)}')
print(f'   Modelo: {metadata["model"]}')
print(f'   Input: {train_config["input_size"]}x{train_config["input_size"]}')
