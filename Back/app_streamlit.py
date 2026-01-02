"""
🌿 AtlasLeaf - Sistema de Diagnóstico de Doenças em Soja
Interface Web com Streamlit
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
from pathlib import Path

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="AtlasLeaf - Diagnóstico de Soja",
    page_icon="🌿",
    layout="centered"
)

# ==================== CARREGAR MODELO E METADADOS ====================
@st.cache_resource
def load_model():
    """Carrega o modelo ONNX e metadados (cached para performance)"""
    base_dir = Path(__file__).parent
    
    # Carregar metadados
    metadata_path = base_dir / "atlasleaf_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Carregar modelo ONNX
    model_path = base_dir / "atlasleaf_soybean.onnx"
    session = ort.InferenceSession(str(model_path))
    
    return session, metadata

# ==================== FUNÇÃO DE PRÉ-PROCESSAMENTO ====================
def preprocess_image(image: Image.Image, metadata: dict) -> np.ndarray:
    """Pré-processa a imagem exatamente como no treinamento"""
    # Converter para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    img_size = metadata['preprocessing']['resize']
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    
    # Converter para array e normalizar para [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Transpor de HWC para CHW (PyTorch format)
    img_array = img_array.transpose(2, 0, 1)
    
    # Normalizar com mean e std do ImageNet
    mean = np.array(metadata['preprocessing']['mean']).reshape(3, 1, 1)
    std = np.array(metadata['preprocessing']['std']).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

# ==================== FUNÇÃO DE PREDIÇÃO ====================
def predict(session, image_array: np.ndarray, metadata: dict) -> tuple:
    """Executa a predição e retorna classe e confiança"""
    input_name = metadata['input_name']
    
    # Executar inferência
    outputs = session.run(None, {input_name: image_array})
    
    # Aplicar softmax para obter probabilidades
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))  # Estabilidade numérica
    probabilities = exp_logits / exp_logits.sum()
    
    # Obter classe predita e confiança
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx] * 100
    predicted_class = metadata['classes'][predicted_idx]
    
    return predicted_class, confidence, probabilities

# ==================== INTERFACE PRINCIPAL ====================
def main():
    # Header
    st.title("🌿 AtlasLeaf")
    st.markdown("### Sistema de Diagnóstico de Doenças em Soja")
    st.markdown("---")
    
    # Carregar modelo
    try:
        session, metadata = load_model()
        st.success("✅ Modelo carregado com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        st.info("Certifique-se de que os arquivos `atlasleaf_soybean.onnx` e `atlasleaf_metadata.json` estão na pasta.")
        return
    
    # Info do modelo
    with st.expander("ℹ️ Informações do Modelo"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Arquitetura:** {metadata['model_architecture']}")
            st.write(f"**Dataset:** {metadata['dataset']}")
        with col2:
            st.write(f"**Classes:** {', '.join(metadata['classes'])}")
            st.write(f"**Input:** {metadata['input_shape'][2]}x{metadata['input_shape'][3]} pixels")
    
    st.markdown("---")
    
    # Upload de imagem
    st.markdown("### 📤 Envie uma imagem de folha de soja")
    uploaded_file = st.file_uploader(
        "Arraste ou clique para selecionar",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formatos aceitos: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Exibir imagem
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Imagem Enviada")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔬 Diagnóstico")
            
            # Processar e predizer
            with st.spinner("Analisando imagem..."):
                img_array = preprocess_image(image, metadata)
                predicted_class, confidence, probabilities = predict(session, img_array, metadata)
            
            # Exibir resultado
            if predicted_class == "Disease":
                st.error(f"🦠 **{predicted_class.upper()}**")
                st.markdown("A folha apresenta sinais de **doença**.")
            else:
                st.success(f"✅ **{predicted_class.upper()}**")
                st.markdown("A folha está **saudável**.")
            
            # Barra de confiança
            st.markdown(f"**Confiança:** {confidence:.1f}%")
            st.progress(float(confidence / 100))
            
            # Probabilidades detalhadas
            st.markdown("---")
            st.markdown("**Probabilidades por classe:**")
            for i, class_name in enumerate(metadata['classes']):
                prob = probabilities[i] * 100
                st.write(f"- {class_name}: {prob:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🌿 AtlasLeaf - Desenvolvido com IA para agricultura sustentável<br>"
        f"Modelo treinado com {metadata.get('total_images', 'N/A')} imagens do dataset SoyNet"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
