"""
🌿 AtlasLeaf - Detector de Saúde de Folhas de Soja
Versão 1.0: Classificação Binária (Saudável vs Doente)
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
from pathlib import Path

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="AtlasLeaf - Detector de Saúde",
    page_icon="🌿",
    layout="centered"
)

# ==================== CARREGAR MODELO ====================
@st.cache_resource
def load_model():
    """Carrega o modelo ONNX v1.0 (Disease/Healthy)"""
    base_dir = Path(__file__).parent
    
    with open(base_dir / "atlasleaf_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    session = ort.InferenceSession(str(base_dir / "atlasleaf_soybean.onnx"))
    
    return session, metadata

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
def predict(session, image_array: np.ndarray, input_name: str, classes: list) -> dict:
    """Executa predição e retorna resultado"""
    outputs = session.run(None, {input_name: image_array})
    logits = outputs[0][0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Encontrar índices
    disease_idx = classes.index('Disease') if 'Disease' in classes else 0
    healthy_idx = classes.index('Healthy') if 'Healthy' in classes else 1
    
    disease_prob = probabilities[disease_idx] * 100
    healthy_prob = probabilities[healthy_idx] * 100
    
    return {
        'is_healthy': healthy_prob > disease_prob,
        'healthy_prob': healthy_prob,
        'disease_prob': disease_prob,
        'confidence': max(healthy_prob, disease_prob)
    }

# ==================== INTERFACE ====================
def main():
    st.title("🌿 AtlasLeaf")
    st.markdown("### Detector de Saúde em Folhas de Soja")
    st.markdown("*Versão 1.0 - Classificação Binária*")
    st.markdown("---")
    
    # Carregar modelo
    try:
        session, metadata = load_model()
        st.success("✅ Modelo carregado com sucesso!")
    except FileNotFoundError as e:
        st.error(f"❌ Modelo não encontrado: {e}")
        st.info("Certifique-se de que os arquivos `atlasleaf_soybean.onnx` e `atlasleaf_metadata.json` estão na pasta Back/")
        return
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Informações do modelo
    with st.expander("ℹ️ Sobre o modelo"):
        st.markdown(f"""
        **Dataset:** {metadata.get('dataset', 'SoyNet')}  
        **Imagens de treino:** {metadata.get('total_images', '3.720')}  
        **Arquitetura:** {metadata.get('model_architecture', 'ResNet18')}  
        **Classes:** Saudável (Healthy) vs Doente (Disease)
        """)
    
    st.markdown("---")
    
    # Upload de imagem
    st.markdown("### 📤 Envie uma imagem de folha de soja")
    uploaded_file = st.file_uploader(
        "Arraste ou clique para selecionar",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Imagem Enviada")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔬 Resultado da Análise")
            
            with st.spinner("Analisando..."):
                # Pré-processar
                prep = metadata['preprocessing']
                img_array = preprocess_image(
                    image, 
                    prep['resize'], 
                    prep['mean'], 
                    prep['std']
                )
                
                # Predição
                result = predict(
                    session, 
                    img_array, 
                    metadata['input_name'],
                    metadata['classes']
                )
            
            # Exibir resultado
            if result['is_healthy']:
                st.success(f"✅ **Folha Saudável**")
                st.markdown(f"Confiança: **{result['healthy_prob']:.1f}%**")
            else:
                st.error(f"🦠 **Folha com Doença Detectada**")
                st.markdown(f"Confiança: **{result['disease_prob']:.1f}%**")
                st.warning("⚠️ Recomenda-se consultar um agrônomo para diagnóstico específico.")
            
            # Barra de progresso
            st.markdown("---")
            st.markdown("**Probabilidades:**")
            
            col_h, col_d = st.columns(2)
            with col_h:
                st.metric("Saudável", f"{result['healthy_prob']:.1f}%")
            with col_d:
                st.metric("Doente", f"{result['disease_prob']:.1f}%")
            
            # Barra visual
            if result['is_healthy']:
                st.progress(float(result['healthy_prob'] / 100))
            else:
                st.progress(float(result['disease_prob'] / 100))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🌿 AtlasLeaf v1.0<br>"
        "Treinado com SoyNet Dataset (3.720 imagens)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
