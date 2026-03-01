"""Script rápido para exportar modelo treinado para ONNX"""
import torch
from pathlib import Path
from data_pipeline.model_v31 import create_model
from data_pipeline.config_m3pro import M3ProConfig

config = M3ProConfig()
BACK_DIR = Path(__file__).parent

print("🔄 Carregando modelo treinado...")
model = create_model(
    model_name=config.model_name,
    num_classes=config.num_classes,
    pretrained=False,  # Vamos carregar pesos do checkpoint
    dropout=config.dropout
)
checkpoint = torch.load(BACK_DIR / 'atlasleaf_v31_best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("🔄 Exportando para ONNX...")
dummy_input = torch.randn(1, 3, config.input_size, config.input_size)
torch.onnx.export(
    model,
    dummy_input,
    BACK_DIR / 'atlasleaf_v31_diseases.onnx',
    export_params=True,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    dynamo=False,
)
print('✅ Modelo exportado: atlasleaf_v31_diseases.onnx')
