"""
🌿 AtlasLeaf v3.1 - Interface com TTA e Detecção de Incerteza

Melhorias:
- Test-Time Augmentation (TTA) para maior precisão
- Detecção de incerteza
- Recomendações baseadas em confiança
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Importa pipeline de inferência v3.1
try:
    from data_pipeline.inference_v31 import AtlasLeafInference, UncertaintyDetector
    from data_pipeline.model_v31 import create_model
    import torch
    V31_AVAILABLE = True
except ImportError:
    V31_AVAILABLE = False


# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="AtlasLeaf v3.1 - Diagnóstico Avançado",
    page_icon="🌿",
    layout="centered"
)

# ==================== CONSTANTES ====================
SEVERITY_INFO = {
    "critical": {"emoji": "🔴", "label": "CRÍTICA", "color": "#dc3545", "action": "Ação imediata necessária!"},
    "high": {"emoji": "🟠", "label": "ALTA", "color": "#fd7e14", "action": "Tratamento recomendado em 48h"},
    "medium": {"emoji": "🟡", "label": "MODERADA", "color": "#ffc107", "action": "Monitorar e tratar se necessário"},
    "low": {"emoji": "🟢", "label": "BAIXA", "color": "#28a745", "action": "Baixo risco, observar evolução"},
    "none": {"emoji": "✅", "label": "SAUDÁVEL", "color": "#20c997", "action": "Nenhuma ação necessária"},
}

# ==================== CARREGAR MODELO ====================
@st.cache_resource
def load_model():
    """Carrega modelo ONNX e metadados."""
    base_dir = Path(__file__).parent
    
    # Procura por v3.1 primeiro, depois v3.0
    for version in ["v31", "v3"]:
        onnx_path = base_dir / f"atlasleaf_{version}_diseases.onnx"
        meta_path = base_dir / f"atlasleaf_{version}_metadata.json"
        
        if onnx_path.exists() and meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            session = ort.InferenceSession(str(onnx_path))
            return session, metadata, version
    
    raise FileNotFoundError("Modelo não encontrado. Execute o treinamento primeiro.")


def load_pytorch_model():
    """Carrega modelo PyTorch diretamente (para TTA)."""
    base_dir = Path(__file__).parent
    
    model_path = base_dir / "atlasleaf_v31_best_model.pth"
    if not model_path.exists():
        return None, None
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    from data_pipeline.model_v31 import create_model
    model = create_model(
        model_name=checkpoint.get('config', {}).get('model_name', 'efficientnet_b3'),
        num_classes=checkpoint.get('config', {}).get('num_classes', 15),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


# ==================== PRÉ-PROCESSAMENTO ====================
def preprocess_image(image: Image.Image, metadata: dict) -> np.ndarray:
    """Pré-processa a imagem para inferência."""
    prep = metadata['preprocessing']
    img_size = prep['resize']
    mean = np.array(prep['mean']).reshape(3, 1, 1)
    std = np.array(prep['std']).reshape(3, 1, 1)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = (img_array - mean) / std
    
    return np.expand_dims(img_array, axis=0).astype(np.float32)


# ==================== INFERÊNCIA COM TTA ====================
def predict_with_tta(image: Image.Image, metadata: dict) -> dict:
    """Faz predição usando TTA se disponível."""
    
    if not V31_AVAILABLE:
        # Fallback para predição simples
        return predict_simple(image, metadata)
    
    # Tenta carregar modelo PyTorch para TTA
    model, checkpoint = load_pytorch_model()
    
    if model is None:
        st.info("ℹ️ Modelo PyTorch não encontrado. Usando inferência ONNX (sem TTA).")
        return predict_simple(image, metadata)
    
    # Usa pipeline v3.1 com TTA
    device = torch.device('cpu')
    pipeline = AtlasLeafInference(model, device, use_tta=True)
    
    # Converte PIL para tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predição com TTA
    with torch.no_grad():
        result = pipeline.predict(image, return_details=False)
    
    # Formata resultado
    classes = metadata['classes']
    probs = result['all_probabilities']
    
    results = []
    for i, prob in enumerate(probs):
        if i < len(classes):
            class_info = classes[i]
            results.append({
                'id': i,
                'name': class_info['name'],
                'friendly_name': class_info['friendly_name'],
                'scientific_name': class_info.get('scientific_name', ''),
                'severity': class_info.get('severity', 'medium'),
                'probability': prob * 100,
            })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'results': results,
        'is_uncertain': result['is_uncertain'],
        'recommendation': result['recommendation'],
        'confidence': result['confidence'],
    }


def predict_simple(image: Image.Image, metadata: dict) -> dict:
    """Predição simples sem TTA (fallback)."""
    session, _, _ = load_model()
    
    img_array = preprocess_image(image, metadata)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    logits = outputs[0][0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Formata resultado
    classes = metadata['classes']
    results = []
    
    for i, prob in enumerate(probs):
        if i < len(classes):
            class_info = classes[i]
            results.append({
                'id': i,
                'name': class_info['name'],
                'friendly_name': class_info['friendly_name'],
                'scientific_name': class_info.get('scientific_name', ''),
                'severity': class_info.get('severity', 'medium'),
                'probability': float(prob * 100),
            })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    # Detecção simples de incerteza
    top1_conf = results[0]['probability'] / 100
    is_uncertain = top1_conf < 0.7
    
    return {
        'results': results,
        'is_uncertain': is_uncertain,
        'recommendation': "Confiança moderada - verifique visualmente" if is_uncertain else "Predição confiável",
        'confidence': top1_conf,
    }


# ==================== INTERFACE ====================
def render_result_card(result: dict, is_top: bool = False, is_uncertain: bool = False):
    """Renderiza card de resultado."""
    severity = result['severity']
    sev_info = SEVERITY_INFO.get(severity, SEVERITY_INFO['medium'])
    
    # Ajusta cor se incerto
    border_color = sev_info['color']
    if is_uncertain and is_top:
        border_color = "#6c757d"  # Cinza para incerto
    
    if is_top:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {sev_info['color']}22, {sev_info['color']}11);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid {border_color};
            margin-bottom: 20px;
        ">
            <h2 style="margin: 0 0 10px 0;">{sev_info['emoji']} {result['friendly_name']}</h2>
            <p style="margin: 5px 0; color: #666;"><em>{result['scientific_name']}</em></p>
            <p style="margin: 10px 0;">
                <strong>Severidade:</strong> 
                <span style="color: {sev_info['color']}; font-weight: bold;">{sev_info['label']}</span>
            </p>
            <p style="margin: 10px 0;">
                <strong>Confiança:</strong> {result['probability']:.1f}%
            </p>
            {f'<p style="margin: 10px 0; color: #6c757d;">⚠️ {sev_info["action"]}</p>' if severity != 'none' else ''}
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(float(result['probability'] / 100))
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{sev_info['emoji']} {result['friendly_name']}")
        with col2:
            st.write(f"{result['probability']:.1f}%")


def main():
    st.title("🌿 AtlasLeaf v3.1")
    st.markdown("### Diagnóstico Avançado de Doenças da Soja")
    st.markdown("*Com Test-Time Augmentation (TTA) e detecção de incerteza*")
    st.markdown("---")
    
    # Carregar modelo
    try:
        session, metadata, version = load_model()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Modelo carregado: {metadata['model']} (v{version})")
        with col2:
            tta_status = "🟢 TTA Ativo" if V31_AVAILABLE else "🟡 TTA Indisponível"
            st.info(tta_status)
        
        with st.expander("ℹ️ Informações do Modelo"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Versão:** {metadata['version']}")
                st.write(f"**Arquitetura:** {metadata['model']}")
            with col2:
                st.write(f"**Classes:** {metadata['num_classes']}")
                metrics = metadata.get('metrics', {})
                st.write(f"**Acurácia:** {metrics.get('test_accuracy', 0):.1f}%")
            with col3:
                st.write(f"**Dataset:** {metadata.get('dataset', 'N/A')}")
                st.write(f"**Imagens:** {metadata.get('total_images', 'N/A'):,}")
    
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        return
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return
    
    st.markdown("---")
    
    # Configurações TTA
    with st.expander("⚙️ Configurações Avançadas"):
        use_tta = st.checkbox("Usar Test-Time Augmentation (TTA)", value=True, disabled=not V31_AVAILABLE)
        show_details = st.checkbox("Mostrar detalhes técnicos", value=False)
    
    # Upload de imagem
    st.markdown("### 📤 Envie uma imagem de folha de soja")
    
    uploaded_file = st.file_uploader(
        "Arraste ou clique para selecionar",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    use_camera = st.checkbox("📷 Usar câmera")
    camera_image = None
    if use_camera:
        camera_image = st.camera_input("Tire uma foto da folha")
    
    # Processar imagem
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_image is not None:
        image = Image.open(camera_image)
    
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Imagem")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔬 Diagnóstico")
            
            with st.spinner("Analisando com TTA..." if use_tta else "Analisando..."):
                try:
                    if use_tta and V31_AVAILABLE:
                        result = predict_with_tta(image, metadata)
                    else:
                        result = predict_simple(image, metadata)
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
                    result = predict_simple(image, metadata)
            
            results = result['results']
            top_result = results[0]
            
            # Alerta de incerteza
            if result['is_uncertain']:
                st.warning(f"⚠️ {result['recommendation']}")
            
            # Resultado principal
            render_result_card(top_result, is_top=True, is_uncertain=result['is_uncertain'])
            
            # Alerta de severidade
            if top_result['severity'] in ['critical', 'high'] and not result['is_uncertain']:
                st.error(f"🚨 {SEVERITY_INFO[top_result['severity']]['action']}")
            
            # Confiança da predição
            if result['confidence'] < 0.8:
                st.info(f"💡 Dica: {result['recommendation']}")
        
        # Outras possibilidades
        st.markdown("---")
        st.markdown("### 📊 Outras Possibilidades")
        
        for res in results[1:6]:
            if res['probability'] > 1:
                render_result_card(res, is_top=False)
        
        # Detalhes técnicos
        if show_details:
            with st.expander("📈 Detalhes Técnicos"):
                st.markdown("**Todas as probabilidades:**")
                for res in results:
                    if res['probability'] > 0.1:
                        st.write(f"- {res['friendly_name']}: {res['probability']:.2f}%")
                
                st.markdown(f"**Confiança da predição:** {result['confidence']:.3f}")
                st.markdown(f"**Incerto:** {'Sim' if result['is_uncertain'] else 'Não'}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"🌿 AtlasLeaf v{metadata.get('version', '3.1')} | "
        f"Modelo: {metadata.get('model', 'EfficientNet-B3')} | "
        f"TTA: {'Ativo' if use_tta else 'Inativo'}"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
