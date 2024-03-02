import streamlit as st
import numpy as np
import pandas as pd
# from streamlit_option_menu import option_menu
# from datetime import date
import time
# import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_modal import Modal
from streamlit_geolocation import streamlit_geolocation
# import streamlit.components.v1 as components
from math import radians, cos, sin, asin, sqrt
from streamlit_lottie import st_lottie
from streamlit_folium import folium
from streamlit_folium import st_folium


# SET PAGE CONFIGURATION
st.set_page_config(
    page_title= "KPI PERFROMANCE",
    page_icon="📚",
    layout="wide",
)


with open('C:/Users/NJV/Desktop/Project_Course/FoodDelivery/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)
    # st.button(f'<style>{f.read()}</style')


st.markdown("<h1 style = 'text-align: center;'>FOOD RECOMMENDER</h1>", unsafe_allow_html=True)



def GetTopRecmt(dict, df, from_index, to_index):
    # # Get the top-down similarity recommend
    sorted_cos_simi = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    top_recmt = [index for index, _ in sorted_cos_simi[from_index : to_index]]
    filtered_df = df[df.index.isin(top_recmt)]
    index_order_map = {index: order for order, index in enumerate(top_recmt)}
    filtered_df['index_order'] = filtered_df.index.map(index_order_map)
    recmt_df = filtered_df.sort_values(by='index_order')
    recmt_df.reset_index(inplace=True)
    return recmt_df



def RecmtByUserInput(df_description, text_input, from_index, to_index):
    cos_simi_dict = {}
    for id, description in df_description['food_list'].items():
        # Tokenize and vectorize
        paragraph_tokens = [text_input, description]
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(paragraph_tokens)
        # Extract BoW vectors for each paragraph
        paragraph1_vector = bow_matrix.toarray()[0]
        paragraph2_vector = bow_matrix.toarray()[1] ## Resuse CosineResult
        # Calculate cosine similarity
        cosine_sim = cosine_similarity([paragraph1_vector], [paragraph2_vector])[0][0]
        cos_simi_dict[id] = cosine_sim

    recmt_df = GetTopRecmt(cos_simi_dict, df_description, from_index, to_index)
    return recmt_df



def ShowItemInfo(index, data):
    data.replace({np.nan: '*:red[Chưa có thông tin]*'}, inplace=True)
    name = data['name_res_y'][index]
    img_link = data['img_link'][index]
    lat = data['lat'][index]
    long = data['long'][index]
    info_column = st.columns([1,1.9])
    with info_column[0]:
        st.image(img_link, caption = name)
        # with sub_info_column[0]:
        st.write("Chấm điểm **Giá cả**", ": ", data.loc[index, 'price_score'])
        st.write("Chấm điểm **Vị trí**", ": ", data.loc[index, 'position_score'])
        st.write("Chấm điểm **Chất lượng**", ": ", data.loc[index, 'quality_score'])
        st.write("Chấm điểm **Phục vụ**", ": ", data.loc[index, 'service_score'])
        st.write("Chấm điểm **Không gian**", ": ", data.loc[index, 'space_score'])
        # with sub_info_column[1]:
        st.write("**Giờ mở cửa**", ": ", data.loc[index, 'open_time'])
        st.write("**Lượt xem**", ": ", data.loc[index, 'total_views'])
        st.write("**Lượt bình luận**", ": ", data.loc[index, 'total_comment'])
        st.write("**Quận**", ": ", data.loc[index, 'district'])
        st.write("**Khu vực**", ": ", data.loc[index, 'area'])
        st.write("**Menu**", ": ", data.loc[index, 'menu_list'])
    with info_column[1]:
        st.markdown(f'### {name}')
        st.write("**Địa chỉ**", ": ", data.loc[index, 'address'])
        st.write("**Thể loại**", ": ", data.loc[index, 'res_category'])
        st.write("**Phong cách**", ": ", data.loc[index, 'cuisine_category'])
        st.write("**Đối tượng**", ": ", data.loc[index, 'res_audience'])
        

        st.markdown("**Vị trí chi tiết:**")
        map = folium.Map(location=[lat, long], zoom_start=14)

        folium.Marker(
            [lat, long], width=500, height=700, popup = name, tooltip=name
        ).add_to(map)

        st_folium(
            map,  returned_objects=["last_object_clicked"]
        )


## 1.4 Đưa ra các món ăn tương đồng:
@st.cache_data
def getdata(url):
    file_id = url.split('/')[-2]
    link = f"https://drive.google.com/uc?id={file_id}"
    data = pd.read_csv("ClusterResult.csv")
    df_similarity_result = pd.read_csv(link)

    df_similarity_result.fillna(0,inplace=True)
    result = np.array(df_similarity_result) + np.array(df_similarity_result).transpose()
    result = pd.DataFrame(result)
    return data, result

data, df_similarity_result = getdata('https://drive.google.com/file/d/1qww9h5R6S0li75-izxy-GTJxsAtCB2Sg/view?usp=drive_link')

# st.write(df_similarity_result)





def ShowSimiRecomment(data, curr_item_id, cluster):
    with st.container():
        st.markdown("### :hotdog: :orange[Các món tương tự]")
        index_list = data[data['cluster'] == cluster].index
        top_recommend = df_similarity_result[df_similarity_result.index.isin(index_list)][curr_item_id].nlargest(41).iloc[1:]
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
        # name = self.data['name_res_y'][self.id]
        # img_link = self.data['img_link'][self.id]
        cluster = self.data['cluster'][self.id]
        modal = Modal(title = ":fries: :rainbow[Chi tiết quán ăn]", key="Demo Key", max_width = self.max_width)
        with modal.container():
            ShowItemInfo(self.id, data)
            ShowSimiRecomment(self.data, self.id, cluster)
            # modal.close()
        



def ShowGridOfItem(height_size, width_size, data, key_surffix):
    buttons = {}
    curr_page = 1




    # curr_page = abc.GetCurrentpage()
    for i in range(height_size):
        # row_container = st.container(border = True)
        
        with st.container():
            food_content = st.columns([1] * width_size)
            button_content = st.columns([1] * width_size)
            for j in range(width_size):
                index = curr_page * height_size * i + j
                name = data['name_res_y'][index] 
                short_name = name if len(name) < 35 else name[:35] + "..."

                img_link = data['img_link'][index]
                try:
                    food_content[j].image(img_link, caption = short_name)
                except:
                    food_content[j].image("https://www.foody.vn/Style/images/deli-dish-no-image.png", caption = short_name)
                
                buttons[(i,j)] = button_content[j].button("Xem chi tiết", key = key_surffix + str(i) + str(j))

    for i in range(height_size):
        for j in range(width_size):
            if buttons[(i,j)]:
                id = height_size * i + j
                popup = Popup(id, data)
                popup.Show()

  


class NavigateFunction:

    def __init__(self, curr_page):
        self.curr_page = curr_page
        
    def Behaviour(self, key_surffix):
        with st.container():
            nav_area = st.columns([7,1,1,7])
            for col in nav_area:
                if col == nav_area[1]:
                    with col:
                        x = st.button('*<<*', key = key_surffix + '*<<*', disabled = (self.curr_page == 1))
                        if x:
                            self.curr_page = self.curr_page - 1

                if col == nav_area[2]:
                    with col:
                        x = st.button('*>>*', key = key_surffix + '*>>*')
                        if x:
                            self.curr_page = self.curr_page + 1
        # st.write(self.curr_page)
    def GetCurrentpage(self):
        return self.curr_page





def RecmtByUserLocation(df_location, user_location, from_index, to_index):
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

    recmt_df = GetTopRecmt(distance_dict, df_location, from_index, to_index)
    return recmt_df
    


class Footer:
    def __init__(self, avatar_img,
                 linkedin_link, github_link, facebook_link):
        self.avatar_img = avatar_img
        self.linkedin_link = linkedin_link
        self.github_link = github_link
        self.facebook_link = facebook_link

    def show_page(self):
        for _ in range(10):
            st.write('')
        st.divider()
        # footer = st.container(border = True)
        # with footer:
        e1, e2, e3, e4 = st.columns([0.4, 0.1, 2, 2])
        with e1:
            img_url = self.avatar_img
            st.image(img_url, use_column_width = True)
        with e2:
            st.empty()
        with e3:
            st.markdown('👨‍💻Owner: Trần Quốc Toàn')
            st.markdown('🏠Place:  Ho Chi Minh City')
            st.markdown('📞Phone: +(84)942 557 560')
            st.markdown('✉️Email: quoctoan11102000@gmail.com')
        with e4:
            i1, i2, i3 = st.columns(3)
            with i1:
                image_url = 'https://cdn-icons-png.flaticon.com/256/174/174857.png'
                linkedin_url = self.linkedin_link

                clickable_image_html = f"""
                    <a href="{linkedin_url}" target="_blank">
                        <img src="{image_url}" alt="Clickable Image" width="50">
                    </a>
                """
                st.markdown(clickable_image_html, unsafe_allow_html=True)
                st.write("Linked")

            with i2:
                image_url = 'https://cdn-icons-png.flaticon.com/512/25/25231.png'
                git_url = self.github_link

                clickable_image_html = f"""
                    <a href="{git_url}" target="_blank">
                        <img src="{image_url}" alt="Clickable Image" width="50">
                    </a>
                """
                st.markdown(clickable_image_html, unsafe_allow_html=True)
                st.write("Github")
            with i3:
                image_url = 'https://cdn-icons-png.flaticon.com/512/3536/3536394.png'
                fb_url = self.facebook_link

                clickable_image_html = f"""
                    <a href="{fb_url}" target="_blank">
                        <img src="{image_url}" alt="Clickable Image" width="50">
                    </a>
                """
                st.markdown(clickable_image_html, unsafe_allow_html=True)
                st.write("Facebook")
        st.divider()





## Main

"---"


# 1. Body

## 1.1. Filter Food:
# st.subheader("**Nơi đặt bộ lọc**", divider ='rainbow')
# tab1, tab2, tab3 = st.tabs(["📈 Gần đây", "Người khác cũng ăn", "Đang khuyến mãi"])


## 1.2. Default Recommendation Food:



height_size = 3
width_size = 7

# st.write(user_location)


## Main content

width = 7
defaul_height = 1
main_height = 3


page = st.columns([0.2, 0.7])
with page[0]:
    st_lottie("https://lottie.host/e5ba7863-def0-45d4-ac64-212158c91ea5/GeTNTPruL2.json", key = "asdfsadf")
with page[1]:
    st.header("***Demo Project Food recommendation***")
    st.subheader("Chức năng chính: ")
    st.markdown(" - Gợi ý nhà hàng quán ăn dựa trên các tiêu chí.")
    st.markdown(" - Tình hình nhà hàng quán ăn ở TPHCM trên nền tảng Foody.")
    st.markdown(" - Chi tiết về dự án.")
    



st.subheader("**Các quán ăn gần đây**", divider ='rainbow')
x = st.columns([4,2,24])
with x[0]:
    st.markdown(":red[Click to get you location: ] ")
with x[1]:
    user_location = streamlit_geolocation()

try:
    ShowGridOfItem(height_size= defaul_height
                , width_size= width
                , data= RecmtByUserLocation(data, user_location, 0, defaul_height * width)
                , key_surffix= "default")
except:
    st.empty()




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




## Filter function:
def GetFilterList(data, filter_col):
    df_split = data[filter_col].apply(lambda x: str(x).split(', '))
    df_split = df_split.explode(filter_col)
    value_counts = df_split.value_counts()
    #Filter value with frequency more than 15
    values_to_replace = value_counts[value_counts > 15].index
    return values_to_replace.tolist()


def GetFilterData(data):
    df = data.copy()
    category_list = ['Chọn hết'] + GetFilterList(data, "res_category")
    category = st.multiselect('Chọn loại hình', options = category_list, default = 'Chọn hết')
    cuisine_list = ['Chọn hết'] + GetFilterList(data, "cuisine_category")
    cuisine = st.multiselect('Chọn nhóm nhỏ', options = cuisine_list, default = 'Chọn hết')
    
    if "Chọn hết" in category and "Chọn hết" not in cuisine:
        df = data[data['cuisine_category'].str.contains('|'.join(cuisine))]
    elif "Chọn hết" not in category and "Chọn hết" in cuisine:
        df = data[data['res_category'].str.contains('|'.join(category))]
    elif "Chọn hết" not in category and "Chọn hết" not in cuisine:
        df = data[(data['res_category'].str.contains('|'.join(category))) | (data['cuisine_category'].str.contains('|'.join(cuisine)))]
    return df



## Use the styled containers
"---"
st.subheader("**Tự chọn quán ăn theo ý thích**", divider ='rainbow') 

input_col = st.columns([3,7])
with input_col[0]:
    st.write("Hôm nay bạn muốn ăn gì?")
    food_input = st.text_input(
        ':red[Hôm nay ăn gì] bạn ơi :panda_face:',
        placeholder = 'Hãy cho tôi biết ý tưởng về món ăn của bạn!...',
        )
with input_col[1]:
    with st.expander("Chọn theo sở thích"):
        data = GetFilterData(data)



if (food_input != ""):
    data = RecmtByUserInput(data, food_input, 0, main_height * width)


   
ShowGridOfItem(height_size = main_height, width_size = width, data = data.reset_index(), key_surffix = "main")


# "---"

footer = Footer(
    'https://scontent.fsgn8-4.fna.fbcdn.net/v/t1.6435-9/148275772_1342911596063325_3059342905885115862_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=be3454&_nc_eui2=AeFxW95d_7EAKV5GOVyKmrTSKR1B5NW0V5gpHUHk1bRXmHza6EBMkV21mu7hTcgTQdKT1QTl1LH_pj4URfYAODsd&_nc_ohc=UiTLjMMEtdoAX_8b7lf&_nc_ht=scontent.fsgn8-4.fna&oh=00_AfB5Hewj1fFwzTcpg24zj20OIVtP75u73w3rf_Pcpb3ISQ&oe=6602AEF8',
    'https://www.linkedin.com/in/toan-tran-555a5621b/',
    'https://github.com/ToanToan110/',
    'https://www.facebook.com/profile.php?id=100010334923606'
)
footer.show_page()
