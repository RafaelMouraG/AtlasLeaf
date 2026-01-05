"""
🌿 AtlasLeaf - Sistema em Cascata
Combina v1.0 (Disease/Healthy) + v2.0 (10 doenças específicas)
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
from pathlib import Path

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="AtlasLeaf - Diagnóstico Cascata",
    page_icon="🌿",
    layout="centered"
)

# ==================== CARREGAR MODELOS ====================
@st.cache_resource
def load_models():
    """Carrega ambos os modelos ONNX"""
    base_dir = Path(__file__).parent
    
    # Modelo v1.0 - Disease/Healthy
    with open(base_dir / "atlasleaf_metadata.json", 'r', encoding='utf-8') as f:
        metadata_v1 = json.load(f)
    session_v1 = ort.InferenceSession(str(base_dir / "atlasleaf_soybean.onnx"))
    
    # Modelo v2.0 - 10 doenças
    with open(base_dir / "atlasleaf_v2_metadata.json", 'r', encoding='utf-8') as f:
        metadata_v2 = json.load(f)
    session_v2 = ort.InferenceSession(str(base_dir / "atlasleaf_v2_diseases.onnx"))
    
    return (session_v1, metadata_v1), (session_v2, metadata_v2)

# ==================== PRÉ-PROCESSAMENTO ====================
def preprocess_image(image: Image.Image, img_size: int, mean: list, std: list) -> np.ndarray:
    """Pré-processa a imagem para inferência"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# ==================== PREDIÇÃO ====================
def predict(session, image_array: np.ndarray, input_name: str) -> tuple:
    """Executa predição e retorna probabilidades"""
    outputs = session.run(None, {input_name: image_array})
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    return probabilities

# ==================== SISTEMA EM CASCATA ====================
def cascade_predict(image: Image.Image, models: tuple) -> dict:
    """
    Sistema em cascata:
    1. v1.0 verifica se está doente
    2. Se doente, v2.0 identifica a doença específica
    """
    (session_v1, meta_v1), (session_v2, meta_v2) = models
    
    # Pré-processar para v1.0
    prep_v1 = meta_v1['preprocessing']
    img_v1 = preprocess_image(image, prep_v1['resize'], prep_v1['mean'], prep_v1['std'])
    
    # ETAPA 1: v1.0 - Disease ou Healthy?
    probs_v1 = predict(session_v1, img_v1, meta_v1['input_name'])
    
    # Encontrar índice de cada classe
    classes_v1 = meta_v1['classes']
    disease_idx = classes_v1.index('Disease') if 'Disease' in classes_v1 else 0
    healthy_idx = classes_v1.index('Healthy') if 'Healthy' in classes_v1 else 1
    
    disease_prob = probs_v1[disease_idx] * 100
    healthy_prob = probs_v1[healthy_idx] * 100
    
    result = {
        'stage1': {
            'is_diseased': disease_prob > healthy_prob,
            'disease_prob': disease_prob,
            'healthy_prob': healthy_prob,
            'confidence': max(disease_prob, healthy_prob)
        },
        'stage2': None
    }
    
    # ETAPA 2: Se doente, identificar qual doença
    if result['stage1']['is_diseased']:
        prep_v2 = meta_v2['preprocessing']
        img_v2 = preprocess_image(image, prep_v2['resize'], prep_v2['mean'], prep_v2['std'])
        
        probs_v2 = predict(session_v2, img_v2, meta_v2['input_name'])
        
        # Top 5 doenças
        sorted_indices = np.argsort(probs_v2)[::-1]
        
        diseases = []
        for idx in sorted_indices:
            cls_name = meta_v2['classes'][idx]
            friendly = meta_v2['friendly_names'].get(cls_name, cls_name)
            diseases.append({
                'name': cls_name,
                'friendly_name': friendly,
                'probability': probs_v2[idx] * 100
            })
        
        result['stage2'] = {
            'top_disease': diseases[0],
            'all_diseases': diseases
        }
    
    return result

# ==================== SEVERIDADE ====================
def get_severity_info(disease_name: str) -> tuple:
    """Retorna emoji e descrição de severidade"""
    severe = {
        'Ferrugem Asiática': ('🔴', 'ALTA', 'Pode causar perdas de 10-80% na produção'),
        'Síndrome da Morte Súbita': ('🔴', 'ALTA', 'Pode matar a planta rapidamente'),
        'Podridão do Colo': ('🔴', 'ALTA', 'Afeta raízes e colo da planta')
    }
    moderate = {
        'Mancha Bacteriana': ('🟠', 'MÉDIA', 'Reduz área fotossintética'),
        'Mancha Marrom': ('🟠', 'MÉDIA', 'Causa desfolha precoce'),
        'Septoriose': ('🟠', 'MÉDIA', 'Afeta folhas inferiores'),
        'Crestamento': ('🟠', 'MÉDIA', 'Queima foliar em condições secas')
    }
    mild = {
        'Vírus do Mosaico': ('🟡', 'BAIXA', 'Reduz vigor da planta'),
        'Mosaico Amarelo': ('🟡', 'BAIXA', 'Causa mosqueado nas folhas'),
        'Oídio': ('🟡', 'BAIXA', 'Fungo superficial, fácil controle')
    }
    
    if disease_name in severe:
        return severe[disease_name]
    elif disease_name in moderate:
        return moderate[disease_name]
    elif disease_name in mild:
        return mild[disease_name]
    return ('⚪', 'DESCONHECIDA', '')

# ==================== INTERFACE ====================
def main():
    st.title("🌿 AtlasLeaf")
    st.markdown("### Sistema de Diagnóstico em Cascata")
    st.markdown("*v1.0 (detecção) + v2.0 (identificação)*")
    st.markdown("---")
    
    # Carregar modelos
    try:
        models = load_models()
        st.success("✅ Modelos v1.0 e v2.0 carregados!")
    except FileNotFoundError as e:
        st.error(f"❌ Modelo não encontrado: {e}")
        return
    except Exception as e:
        st.error(f"❌ Erro: {e}")
        return
    
    # Explicação do sistema
    with st.expander("ℹ️ Como funciona o sistema em cascata?"):
        st.markdown("""
        **Etapa 1 - Detecção (v1.0)**
        - Modelo treinado com 3.720 imagens
        - Determina: Folha SAUDÁVEL ou DOENTE?
        
        **Etapa 2 - Identificação (v2.0)**
        - Só é executada se a folha estiver doente
        - Identifica qual das 10 doenças específicas
        
        **Vantagem:** Maior precisão combinando dois modelos especializados
        """)
    
    st.markdown("---")
    
    # Upload
    st.markdown("### 📤 Envie uma imagem de folha de soja")
    uploaded_file = st.file_uploader(
        "Arraste ou clique para selecionar",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Imagem")
            st.image(image, width="stretch")
        
        with col2:
            st.markdown("#### 🔬 Análise em Cascata")
            
            with st.spinner("Processando..."):
                result = cascade_predict(image, models)
            
            # ETAPA 1
            st.markdown("**Etapa 1: Detecção**")
            stage1 = result['stage1']
            
            if stage1['is_diseased']:
                st.error(f"🦠 Doença detectada ({stage1['disease_prob']:.1f}%)")
            else:
                st.success(f"✅ Folha Saudável ({stage1['healthy_prob']:.1f}%)")
                
        
        # ETAPA 2 (se doente)
        if result['stage2']:
            st.markdown("---")
            st.markdown("### 🔍 Etapa 2: Identificação da Doença")
            
            top = result['stage2']['top_disease']
            emoji, severity, description = get_severity_info(top['friendly_name'])
            
            # Card principal
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b22, #ff8e5322); 
                        padding: 20px; border-radius: 10px; border-left: 5px solid #ff6b6b;">
                <h2>{emoji} {top['friendly_name']}</h2>
                <p><strong>Severidade:</strong> {severity}</p>
                <p><strong>Confiança:</strong> {top['probability']:.1f}%</p>
                <p style="color: gray;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barra de confiança
            st.progress(float(top['probability'] / 100))
            
            if top['probability'] < 50:
                st.warning("⚠️ Confiança baixa - considere consultar um agrônomo")
            
            # Outras possibilidades
            st.markdown("---")
            st.markdown("### 📊 Outras Possibilidades")
            
            other_diseases = result['stage2']['all_diseases'][1:5]
            for disease in other_diseases:
                prob = disease['probability']
                if prob > 5:  # Só mostra se > 5%
                    st.write(f"• {disease['friendly_name']}: {prob:.1f}%")
        
        # Comparação de confiança
        if result['stage2']:
            st.markdown("---")
            with st.expander("📈 Detalhes técnicos"):
                st.markdown("**Modelo v1.0 (Disease/Healthy):**")
                st.write(f"- Doente: {stage1['disease_prob']:.2f}%")
                st.write(f"- Saudável: {stage1['healthy_prob']:.2f}%")
                
                st.markdown("**Modelo v2.0 (10 doenças):**")
                for d in result['stage2']['all_diseases']:
                    st.write(f"- {d['friendly_name']}: {d['probability']:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🌿 AtlasLeaf - Sistema em Cascata<br>"
        "v1.0 (3.720 imgs) + v2.0 (609 imgs)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
