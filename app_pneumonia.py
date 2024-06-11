import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

# Carregar o modelo treinado
model = load_model('modelo_vgg16_pneumonia.h5')

# Função para fazer previsões e mostrar a matriz de confusão
def predict_and_show_confusion_matrix(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_class = np.where(predictions > 0.5, 1, 0).flatten()[0]
    if predicted_class == 0:
        st.write("A imagem foi classificada como NORMAL.")
    else:
        st.write("A imagem foi classificada como PNEUMONIA.")

# Aplicativo Streamlit
def main():
    st.title('Projeto de Joao Coimbra')
    st.title('Modelo de Classificação de Imagens')

    # Widget de upload de arquivo
    uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")
    
    # Botão para fazer previsões
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Imagem carregada', use_column_width=True)
        if st.button('Fazer Previsões'):
            predict_and_show_confusion_matrix(img)

if __name__ == '__main__':
    main()

