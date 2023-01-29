import numpy as np
import streamlit as st
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img
from PIL import Image

model = load_model('./Models/vgg16.h5',compile=False)
lab = {0: 'ABBOTTS BABBLER', 1: 'ABBOTTS BOOBY', 2: 'ABYSSINIAN GROUND HORNBILL', 3: 'AFRICAN CROWNED CRANE', 4: 'AFRICAN EMERALD CUCKOO', 5: 'AFRICAN FIREFINCH', 6: 'AFRICAN OYSTER CATCHER', 7: 'AFRICAN PIED HORNBILL', 8: 'ALBATROSS', 9: 'ALBERTS TOWHEE', 10: 'ALEXANDRINE PARAKEET', 11: 'ALPINE CHOUGH', 12: 'ALTAMIRA YELLOWTHROAT', 13: 'AMERICAN AVOCET', 14: 'AMERICAN BITTERN', 15: 'AMERICAN COOT', 16: 'AMERICAN FLAMINGO', 17: 'AMERICAN GOLDFINCH', 18: 'AMERICAN KESTREL', 19: 'AMERICAN PIPIT', 20: 'UMBRELLA BIRD', 21: 'VARIED THRUSH', 22: 'VEERY', 23: 'VENEZUELIAN TROUPIAL', 24: 'VERMILION FLYCATHER', 25: 'VICTORIA CROWNED PIGEON', 26: 'VIOLET GREEN SWALLOW', 27: 'VIOLET TURACO', 28: 'VULTURINE GUINEAFOWL', 29: 'WALL CREAPER', 30: 'WATTLED CURASSOW', 31: 'WATTLED LAPWING', 32: 'WHIMBREL', 33: 'WHITE BROWED CRAKE', 34: 'WHITE CHEEKED TURACO', 35: 'WHITE CRESTED HORNBILL', 36: 'WHITE NECKED RAVEN', 37: 'WHITE TAILED TROPIC', 38: 'WHITE THROATED BEE EATER', 39: 'WILD TURKEY', 40: 'WILSONS BIRD OF PARADISE', 41: 'WOOD DUCK', 42: 'YELLOW BELLIED FLOWERPECKER', 43: 'YELLOW CACIQUE', 44: 'YELLOW HEADED BLACKBIRD'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Birds Species Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "45 Bird Species"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Bird is: "+result)
run()