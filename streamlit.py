import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import date
import time
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_modal import Modal
from streamlit_geolocation import streamlit_geolocation
import streamlit.components.v1 as components
from math import radians, cos, sin, asin, sqrt

# SET PAGE CONFIGURATION
st.set_page_config(
    page_title= "KPI PERFROMANCE",
    page_icon="📚",
    layout="wide",
)

def GetTopRecmt(dict, df, no_of_recmt):
    # # Get the top-down similarity recommend
    sorted_cos_simi = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    top_recmt = [index for index, _ in sorted_cos_simi[:no_of_recmt]]
    filtered_df = df[df.index.isin(top_recmt)]
    index_order_map = {index: order for order, index in enumerate(top_recmt)}
    filtered_df['index_order'] = filtered_df.index.map(index_order_map)
    recmt_df = filtered_df.sort_values(by='index_order')
    recmt_df.reset_index(inplace=True)
    return recmt_df



def RecmtByUserInput(df_description, text_input, no_of_recmt):
    cos_simi_dict = {}
    for id, description in df_description['food_list'].items():
        # Tokenize and vectorize
        paragraph_tokens = [text_input, description]
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(paragraph_tokens)
        # Extract BoW vectors for each paragraph
        paragraph1_vector = bow_matrix.toarray()[0]
        paragraph2_vector = bow_matrix.toarray()[1]
        # Calculate cosine similarity
        cosine_sim = cosine_similarity([paragraph1_vector], [paragraph2_vector])[0][0]
        cos_simi_dict[id] = cosine_sim

    recmt_df = GetTopRecmt(cos_simi_dict, df_description, no_of_recmt)
    return recmt_df



def ShowItemInfo(index, img_link, name):
    info_column = st.columns([1,2])
    with info_column[0]:
        st.image(img_link, caption = name)
        sub_info_column = st.columns([0.5,1])
        with sub_info_column[0]:
            st.write("Giá cả", ": ", data.loc[index, 'price_score'])
            st.write("Vị trí", ": ", data.loc[index, 'position_score'])
            st.write("Chất lượng", ": ", data.loc[index, 'quality_score'])
            st.write("Phục vụ", ": ", data.loc[index, 'service_score'])
            st.write("Không gian", ": ", data.loc[index, 'space_score'])
        with sub_info_column[1]:
            st.write("Giờ mở cửa", ": ", data.loc[index, 'open_time'])
            st.write("Lượt xem", ": ", data.loc[index, 'total_views'])
            st.write("Lượt bình luận", ": ", data.loc[index, 'total_comment'])
            st.write("Quận", ": ", data.loc[index, 'district'])
            st.write("Khu vực", ": ", data.loc[index, 'area'])
    with info_column[1]:
        st.markdown(f'### {name}')
        st.write("Địa chỉ", ": ", data.loc[index, 'address'])
        st.write("Thể loại", ": ", data.loc[index, 'res_category'])
        st.write("Phong cách", ": ", data.loc[index, 'cuisine_category'])
        st.write("Đối tượng", ": ", data.loc[index, 'res_audience'])
        st.write("Menu", ": ", data.loc[index, 'menu_list'])


## 1.4 Đưa ra các món ăn tương đồng:
@st.cache_data
def getdata():
    df_similarity_result = pd.read_csv('C:/Users/NJV/Desktop/FoodDelivery/CosineSimResult.csv')
    return df_similarity_result
df_similarity_result = getdata()

def ShowSimiRecomment(id, cluster):
    with st.container(height = 1000):
        st.markdown("## Các món tương tự")
        index_list = data[data['cluster'] == cluster].index
        top_recommend = df_similarity_result[df_similarity_result.index.isin(index_list)][id].nlargest(51).iloc[1:]
        top_recommend_id = top_recommend.index
        # st.write(recommendw_with_cosin_sim)
        names = data[data['name_res_y'].index.isin(top_recommend_id)]['name_res_y'].tolist()
        links = data[data['name_res_y'].index.isin(top_recommend_id)]['img_link'].tolist()

        for name,link,index in zip(names,links,top_recommend_id):
            info_column = st.columns([1,2])
            with info_column[0]:
                try:
                    st.image(link, caption = name)
                except:
                    continue
            with info_column[1]:
                st.markdown(f'#### {name}')
                st.write("Địa chỉ", ": ", data.loc[index, 'address'])
                st.write("Thể loại", ": ", data.loc[index, 'res_category'])
                st.write("Phong cách", ": ", data.loc[index, 'cuisine_category'])
                st.write("Đối tượng", ": ", data.loc[index, 'res_audience'])



class Popup:
    def __init__(self, id, data, max_width = 1000):
        self.data = data
        self.max_width = max_width
        self.id = id

    def Show(self):
        name = data['name_res_y'][self.id]
        img_link = data['img_link'][self.id]
        cluster = data['cluster'][self.id]
        modal = Modal(title = "Chi tiết quán ăn", key="Demo Key", max_width = self.max_width)
        with modal.container():
            ShowItemInfo(self.id, img_link, name)
            ShowSimiRecomment(str(self.id), cluster)
            # modal.close()
        



def ShowGridOfItem(height_size, width_size, data, key_surffix):
    buttons = {}
    for i in range(height_size):
        row_container = st.container(border = True)
        food_content = st.columns([1] * width_size)
        with row_container:
            for j in range(width_size):
                index = height_size * i + j
                name = data['name_res_y'][index]
                img_link = data['img_link'][index]

                try:
                    food_content[j].image(img_link, caption = name)
                except:
                    food_content[j].image("https://www.foody.vn/Style/images/deli-dish-no-image.png", caption = name)

                buttons[(i,j)] = food_content[j].button(key_surffix + str(i) + str(j), key = key_surffix + str(i) + str(j))

    for i in range(height_size):
        for j in range(width_size):
            if buttons[(i,j)]:
                id = height_size * i + j
                popup = Popup(id, data)
                popup.Show()



def RecmtByUserLocation(df_location, user_location, number_of_recmt):
    distance_dict = {}
    lat1 =  user_location['latitude']
    lon1 = user_location['longitude']

    # Calculate the distance between user to each restaurant
    for id in df_location.index:
        location = df_location.iloc[id,:]
        lat2 = location['lat']
        lon2 = location['long']
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # Apply haversine formula to calculate distance
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 
        distance_dict[id] = c*r

    recmt_df = GetTopRecmt(distance_dict, df_location, number_of_recmt)
    return recmt_df
    



## Main

"---"


# 1. Body

## 1.1. Filter Food:
# st.subheader("**Nơi đặt bộ lọc**", divider ='rainbow')
# tab1, tab2, tab3 = st.tabs(["📈 Gần đây", "Người khác cũng ăn", "Đang khuyến mãi"])


## 1.2. Default Recommendation Food:

@st.cache_data
def clusterdata():
    data = pd.read_csv("C:/Users/NJV/Desktop/FoodDelivery/ClusterResult.csv")
    return data
data = clusterdata()


height_size = 3
width_size = 7
user_location = streamlit_geolocation()
# location['latitude']


## Main content

width = 7
defaul_height = 1
main_height = 3

st.subheader("**Nơi Gợi ý theo khoảng cách**", divider ='rainbow')
ShowGridOfItem(height_size= defaul_height
               , width_size= width
               , data= RecmtByUserLocation(data, user_location, defaul_height * width)
               , key_surffix= "default")


st.subheader("**Flash Sales hôm nay**", divider ='rainbow')
ShowGridOfItem(height_size= defaul_height
               , width_size= width
               , data= data[data['promotion'].str.contains("Flash Sale")].reset_index()
               , key_surffix= "promotion_list")

st.subheader("**Người khác cũng ăn**", divider ='rainbow')
ShowGridOfItem(height_size= defaul_height
               , width_size= width
               , data= data.sort_values(by='total_views').reset_index()
               , key_surffix= "other_user")



st.subheader("**Nơi Gợi ý mặc định**", divider ='rainbow')
container1 = st.container(border = True)

## Use the styled containers

container1.write("What do you want to eat today?")
food_input = container1.text_input(
    ':red[Hôm nay ăn gì] bạn ơi :panda_face:',
    placeholder = 'Hãy cho tôi biết ý tưởng về món ăn của bạn!...',
    )

if (food_input != ""):
    data = RecmtByUserInput(data, food_input, main_height * width)
    
ShowGridOfItem(height_size = main_height, width_size = width, data = data, key_surffix = "main")

