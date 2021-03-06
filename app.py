import streamlit as st
from PIL import Image
import style
import os
from io import BytesIO
import base64

st.set_page_config(page_title="Digital Art - Style Transfer",
                   page_icon="./images/brush.png", layout="centered")

title = '<p style="text-align: center; color: White;font-size: 50px;font-weight: 350;font-family:Cursive "> ' \
        'DigitalART </p> '
st.markdown(title, unsafe_allow_html=True)
st.markdown(
    "<b> <i> Создавайте цифровое искусство с помощью нейронной сети!", unsafe_allow_html=True
)

# пути к изображениям стилей:
root_style = "./images/style-images"

# функция загрузки изображения
def get_image_download_link(img, file_name, style_name):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a style = "color:black" href="data:file/jpg;base64,{img_str}" download="{style_name + "_" + file_name + ".jpg"}"><input type="button" value="Cкачать"></a>'
    return href


# создание боковой панели для выбора стиля изображения
st.sidebar.image(image="./images/speed-brush.gif")
st.markdown("</br>", unsafe_allow_html=True)
style_name = st.sidebar.selectbox(
    'Выберите стиль:',
    ("candy", "mosaic", "rain_princess",
     "udnie", "tg", "demon_slayer", "ben_giles", "ben_giles_2", "brush_stroke", "wave", "old_quarter", "green_swirly"))
path_style = os.path.join(root_style, style_name + ".jpg")

# Функция загрузки изображения
img = None
uploaded_file = st.file_uploader(
    "Выберите изображение...", type=["jpg", "jpeg", "png"])
show_file = st.empty()

# проверка файла
if not uploaded_file:
    show_file.info("Файл не загружен")
else:
    img = Image.open(uploaded_file)
    # здесь проверка файла, если файл является файлом изображения
    st.image(img, caption='Загруженное изображение', use_column_width=True)
    st.image(path_style, caption='Стиль изображения', use_column_width=True)

extensions = [".png", ".jpeg", ".jpg"]

if uploaded_file is not None and any(extension in uploaded_file.name for extension in extensions):

    name_file = uploaded_file.name.split(".")
    root_model = "./saved_models"
    model_path = os.path.join(root_model, style_name + ".pth")

    img = img.convert('RGB')
    input_image = img

    root_output = "./images/output-images"
    output_image = os.path.join(
        root_output, style_name + "-" + name_file[0] + ".jpg")

    stylize_button = st.button("Стилизовать")
    if stylize_button:
        model = style.load_model(model_path)
        stylized = style.stylize(model, input_image, output_image)
        # отображение выходного изображения
        st.write("### Выходное изображение")
        # изображение = изображение.открыть выходит (выходное_изображение)
        st.image(stylized, width=500, use_column_width=True)
        st.markdown(get_image_download_link(
            stylized, name_file[0], style_name), unsafe_allow_html=True)
st.balloons()