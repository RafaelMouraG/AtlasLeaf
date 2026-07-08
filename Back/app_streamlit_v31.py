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
except ImportError as e:
    V31_AVAILABLE = False
    print(f"⚠️  Pipeline v3.1 não disponível: {e}")


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
    
    # Prefere o modelo de campo (7 classes), depois v3.1, depois v3.0
    for version in ["field7", "v31", "v3"]:
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
    
    # Prefere o modelo de campo bom (field7); fallbacks p/ os antigos
    for name in ["atlasleaf_field7_best.pth", "atlasleaf_v31_sourcesplit_best.pth",
                 "atlasleaf_v31_best_model.pth"]:
        model_path = base_dir / name
        if model_path.exists():
            break
    else:
        return None, None
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    from data_pipeline.model_v31 import create_model
    model = create_model(
        model_name=checkpoint.get('config', {}).get('model_name', 'efficientnet_v2_s'),
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

    # Recorte de folha (mesmo do treino) se ativado na metadata
    if prep.get('leaf_crop'):
        try:
            from data_pipeline.leaf_segmentation import LeafCropper
            image = LeafCropper(mask_background=prep.get('leaf_crop_mask_bg', False))(image)
        except Exception:
            pass

    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = (img_array - mean) / std
    
    return np.expand_dims(img_array, axis=0).astype(np.float32)


# ==================== FORMATAÇÃO / FILTRO DE CLASSES ====================
def format_results(probs, metadata: dict) -> dict:
    """
    Constrói a lista de resultados considerando SÓ as classes suportadas
    (metadata['supported_class_ids']) e renormaliza as probabilidades entre elas.
    Aplica o limiar de confiança (metadata['confidence_threshold']).
    """
    classes = metadata['classes']
    by_id = {int(c['id']): c for c in classes}
    supported = metadata.get('supported_class_ids')
    if not supported:
        supported = [int(c['id']) for c in classes]  # retrocompatível: usa todas

    # Renormaliza as probabilidades entre as classes suportadas
    sup_probs = {cid: float(probs[cid]) for cid in supported if cid < len(probs)}
    total = sum(sup_probs.values()) or 1.0

    results = []
    for cid, p in sup_probs.items():
        info = by_id.get(cid, {})
        results.append({
            'id': cid,
            'name': info.get('name', str(cid)),
            'friendly_name': info.get('friendly_name', str(cid)),
            'scientific_name': info.get('scientific_name', ''),
            'severity': info.get('severity', 'medium'),
            'probability': 100.0 * p / total,
        })
    results.sort(key=lambda x: x['probability'], reverse=True)

    threshold = float(metadata.get('confidence_threshold', 0.7))
    top1 = results[0]['probability'] / 100 if results else 0.0
    is_uncertain = top1 < threshold
    if is_uncertain:
        rec = f"⚠️ BAIXA CONFIANÇA (<{threshold*100:.0f}%) — verifique visualmente ou capture nova foto"
    else:
        rec = "✅ Predição confiável"
    return {'results': results, 'is_uncertain': is_uncertain,
            'recommendation': rec, 'confidence': top1}


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
    prep = metadata.get('preprocessing', {})
    pipeline = AtlasLeafInference(
        model, device, use_tta=True,
        leaf_crop=prep.get('leaf_crop', False),
        leaf_crop_mask_bg=prep.get('leaf_crop_mask_bg', False),
    )
    
    # CORREÇÃO: Passa a imagem PIL diretamente para a pipeline
    # A pipeline aplica as transformações TTA internamente
    result = pipeline.predict(image, return_details=False)

    # Formata considerando só as classes suportadas + limiar de confiança
    return format_results(result['all_probabilities'], metadata)


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

    # Formata considerando só as classes suportadas + limiar de confiança
    return format_results(probs, metadata)


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
                n_sup = len(metadata.get('supported_class_ids') or metadata.get('classes', []))
                st.write(f"**Classes:** {n_sup}")
                metrics = metadata.get('metrics', {})
                acc = metrics.get('test_accuracy')
                if acc is None:
                    bal = metadata.get('evaluation', {}).get('test_balanced_acc')
                    acc = bal * 100 if bal is not None else None
                st.write(f"**Acurácia:** {acc:.1f}%" if acc is not None else "**Acurácia:** N/A")
            with col3:
                st.write(f"**Dataset:** {metadata.get('dataset', 'N/A')}")
                total = metadata.get('total_images')
                st.write(f"**Imagens:** {total:,}" if isinstance(total, (int, float)) else "**Imagens:** N/A")
    
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
            st.image(image, width="stretch")
        
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
