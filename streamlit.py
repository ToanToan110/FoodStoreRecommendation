import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
# from datetime import date
import time
# import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_modal import Modal
# from streamlit_geolocation import streamlit_geolocation
# import streamlit.components.v1 as components
from math import radians, cos, sin, asin, sqrt
from streamlit_lottie import st_lottie
from streamlit_folium import folium, st_folium
# from streamlit_folium import st_folium
import graphviz
from streamlit_js_eval import get_geolocation
import math
from unidecode import unidecode

# SET PAGE CONFIGURATION
st.set_page_config(
    page_title= "Food Recommender",
    page_icon="🍔",
    layout="wide",
)


with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)
    # st.button(f'<style>{f.read()}</style')




## 1.4 Đưa ra các món ăn tương đồng:
@st.cache_data
def getdata(url):
    file_id = url.split('/')[-2]
    link = f"https://drive.google.com/uc?id={file_id}"
    data = pd.read_csv("ClusterResult.csv")
    # data.drop(['level_0'], axis = 1, inplace=True)
    df_similarity_result = pd.read_csv(link)

    df_similarity_result.fillna(0,inplace=True)
    result = np.array(df_similarity_result) + np.array(df_similarity_result).transpose()
    result = pd.DataFrame(result)
    return data, result

data, df_similarity_result = getdata('https://drive.google.com/file/d/1qww9h5R6S0li75-izxy-GTJxsAtCB2Sg/view?usp=drive_link')


# @st.cache_data
current_location = get_geolocation()['coords']



st.markdown("<h1 style = 'text-align: center;'>FOOD RECOMMENDER</h1>", unsafe_allow_html=True)


@st.cache_data
def GetTopRecmt(dict, df, from_index, to_index, reverse):
    # # Get the top-down similarity recommend
    sorted_cos_simi = sorted(dict.items(), key=lambda x: x[1], reverse=reverse)
    top_recmt = [index for index, _ in sorted_cos_simi]
    filtered_df = df[df.index.isin(top_recmt)]
    index_order_map = {index: order for order, index in enumerate(top_recmt)}
    filtered_df['index_order'] = filtered_df.index.map(index_order_map)
    recmt_df = filtered_df.sort_values(by='index_order')

    recmt_df.reset_index(inplace=True)
    return recmt_df



vectorizer = CountVectorizer()



# @st.cache_data
# def CosineVector(df_description):
#     cos_simi_list = []
#     for id, description in df_description['food_list'].items():
#         # Tokenize and vectorize
#         paragraph_tokens = [description]

#         bow_matrix = vectorizer.fit_transform(paragraph_tokens)
#         # Extract BoW vectors for each paragraph

#         paragraph1_vector = bow_matrix.toarray()[0]
#         cos_simi_list.append(paragraph1_vector)
#     cos_simi_list = pd.DataFrame(cos_simi_list).fillna(0)
#     return cos_simi_list
# st.write(pd.DataFrame(CosineVector(data)))



@st.cache_data
def RecmtByUserInput(df_description, text_input, from_index, to_index):

    text = text_input.split(' ')
    result = 0
    cos_simi_dict = {}
    for id, descrip in df_description['food_list'].items():
        result = 0
        descrip =  unidecode(descrip).lower()
        for i in text:
            x = descrip.count(unidecode(i).lower())
            result += x
        cos_simi_dict[id] = result/len(descrip)
        
    recmt_df = GetTopRecmt(cos_simi_dict, df_description, from_index, to_index, True)
    return recmt_df


# @st.cache_data
def ShowItemInfo(index, data):
    data = data.replace({np.nan: '*:red[Chưa có thông tin]*'})
    name = data['name_res_y'][index]
    img_link = data['img_link'][index]
    lat = data['lat'][index]
    long = data['long'][index]
    st.markdown(f'### {name}')

    info_column = st.columns([1,1.9])
    with info_column[0]:
        try:
            st.image(img_link, caption = name)
        except:
            st.image("https://images.foody.vn/res/g100007/1000062931/prof/s280x175/file_bde8cd9f-a284-4aa8-a46f-697-c2ed7143-230306112732.jpeg", caption = name)

        # with sub_info_column[0]:
        st.write("Điểm **Giá cả**", ": ", data.loc[index, 'price_score'])
        st.write("Điểm **Vị trí**", ": ", data.loc[index, 'position_score'])
        st.write("Điểm **Chất lượng**", ": ", data.loc[index, 'quality_score'])
        st.write("Điểm **Phục vụ**", ": ", data.loc[index, 'service_score'])
        st.write("Điểm **Không gian**", ": ", data.loc[index, 'space_score'])
        # with sub_info_column[1]:
        st.write("**Giờ mở cửa**", ": ", data.loc[index, 'open_time'])
        st.write("**Lượt xem**", ": ", data.loc[index, 'total_views'])
        st.write("**Lượt bình luận**", ": ", data.loc[index, 'total_comment'])
        st.write("**Quận**", ": ", data.loc[index, 'district'])
        st.write("**Khu vực**", ": ", data.loc[index, 'area'])
        st.write("**Menu**", ": ", data.loc[index, 'menu_list'])
        
        # url = data['url'][index]
        # st.mardown(f'[{name}](%s)' % url)
        st.write("**Thông tin gốc**", ": ", f'[{"Link Foody"}](%s)' % data['url'][index])
        
    with info_column[1]:
        st.write("**Địa chỉ**", ": ", data.loc[index, 'address'])
        st.write("**Thể loại**", ": ", data.loc[index, 'res_category'])
        st.write("**Phong cách**", ": ", data.loc[index, 'cuisine_category'])
        st.write("**Đối tượng**", ": ", data.loc[index, 'res_audience'])
        st.markdown("**Vị trí chi tiết:**")

        curr_lat = current_location['latitude']
        curr_long = current_location['longitude']
        map = folium.Map(location=[curr_lat, curr_long], zoom_start=14)
        
        folium.Marker(
            [lat, long], width=600, height=700, popup = name, tooltip=name
        ).add_to(map)
        folium.Marker(
            [curr_lat, curr_long], width=600, height=700, popup = name, tooltip=name
        ).add_to(map)

        st_folium(
            map,  returned_objects=["last_object_clicked"]
        )




def ShowSimiRecomment(data, curr_item_id, cluster):
    with st.container():
        st.subheader(":hotdog: :orange[Các món tương tự]", divider ='rainbow')
        index_list = data[data['cluster'] == cluster].index
        top_recommend = df_similarity_result[df_similarity_result.index.isin(index_list)][curr_item_id].nlargest(41).iloc[1:]
        top_recommend_id = top_recommend.index
        # st.write(recommendw_with_cosin_sim)
        names = data[data['name_res_y'].index.isin(top_recommend_id)]['name_res_y'].tolist()
        links = data[data['name_res_y'].index.isin(top_recommend_id)]['img_link'].tolist()
        urls = data[data['name_res_y'].index.isin(top_recommend_id)]['url'].tolist()

        for name,link,index,url in zip(names,links,top_recommend_id,urls):
            info_column = st.columns([1,0.03,2])
            with st.container(border = True):
                with info_column[0]:
                    try:
                        st.write(' ')
                        st.image(link, caption = name, use_column_width = 'always')
                    except:
                        st.image("https://images.foody.vn/res/g100007/1000062931/prof/s280x175/file_bde8cd9f-a284-4aa8-a46f-697-c2ed7143-230306112732.jpeg", caption = name, use_column_width = 'always')
                        
                with info_column[2]:
                    stri = f' ([{"Link Foody"}](%s)' % url + ')'
                    st.markdown(f'#### {name + stri}')
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
        cluster = self.data['cluster'][self.id]
        modal = Modal(title = ":fries: :rainbow[Chi tiết quán ăn]", key="Demo Key", max_width = self.max_width)
        with modal.container():
            ShowItemInfo(self.id, self.data)
            ShowSimiRecomment(self.data, self.id, cluster)
            # modal.close()
        

# @st.cache_data
def Behaviour(key_surffix):


    disable = False
    curr_page = key_surffix + "cur_page"
    pre_page = -1
    if curr_page not in st.session_state:
        st.session_state[curr_page] =0
    if (st.session_state.get(curr_page) == 1) | (pre_page==0):
        disable = False
    elif st.session_state.get(curr_page) == 0:
        disable = True



    nav_area = st.columns([7,1,1,7])
    with nav_area[2]:
        if st.session_state.get(key_surffix + '*>>>*', False):
            st.session_state[curr_page] = st.session_state.get(curr_page, 0) + 1
            pre_page = st.session_state.get(curr_page, 0) - 1
            if (st.session_state.get(curr_page) == 1) | (pre_page==0):
                disable = False
            elif st.session_state.get(curr_page) == 0:
                disable = True
        st.button('*>>>*', key = key_surffix + '*>>>*')


    with nav_area[1]:
        if st.session_state.get(key_surffix + '*<<<*', False):
            st.session_state[curr_page] = st.session_state.get(curr_page, 0) - 1
            pre_page = st.session_state.get(curr_page, 0) - 1
            if (st.session_state.get(curr_page) == 1) | (pre_page==0):
                disable = False
            elif st.session_state.get(curr_page) == 0:
                disable = True

        st.button('*<<<*', key = key_surffix + '*<<<*', disabled = disable)
    return st.session_state.get(curr_page, 0)





# @st.cache_data
def ShowGridOfItem(height_size, width_size, data, key_surffix):
    buttons = {}
    curr_page = Behaviour(key_surffix)
    for i in range(height_size):
        
        food_content = st.columns([1] * width_size)
        button_content = st.columns([1] * width_size)

        for j in range(width_size):
            index = curr_page*height_size*width_size + i*width_size + j
            name = data['name_res_y'][index] 
            short_name = name if len(name) < 50 else name[:50] + "..."

            img_link = data['img_link'][index]
            try:
                food_content[j].image(img_link, caption = name, use_column_width = 'always')
            except:
                food_content[j].image("https://www.foody.vn/Style/images/deli-dish-no-image.png", caption = name)
            
            buttons[curr_page*height_size*width_size + i*width_size + j] = button_content[j].button("Xem chi tiết", key = key_surffix + str(index))

    for i in range(height_size):
        for j in range(width_size):
            if buttons[curr_page*height_size*width_size + i*width_size + j]:
                id = curr_page*height_size*width_size + i*width_size + j
                popup = Popup(id, data)
                popup.Show()

  


class NavigateFunction:

    def __init__(self, curr_page):
        self.curr_page = curr_page


    def Behaviour(self, key_surffix):
        disable = False
        if 'pre_page' not in st.session_state:
            st.session_state['pre_page'] =-1
        if 'curr_page' not in st.session_state:
            st.session_state['curr_page'] =0
        if (st.session_state.get('curr_page') == 1) | (st.session_state['pre_page']==0):
            disable = False
            # st.write("disable tại đây")
        elif st.session_state.get('curr_page') == 0:
            disable = True

        # with st.container():
        nav_area = st.columns([7,1,1,7])
        with nav_area[2]:
            if st.session_state.get(key_surffix + '*>>*', False):
                
                st.session_state['curr_page'] = st.session_state.get('curr_page', 0) + 1
                st.session_state['pre_page'] = st.session_state.get('curr_page', 0) - 1
                if (st.session_state.get('curr_page') == 1) | (st.session_state['pre_page']==0):
                    disable = False
                elif st.session_state.get('curr_page') == 0:
                    disable = True
            st.button('*>>*', key = key_surffix + '*>>*')

        with nav_area[1]:
            if st.session_state.get(key_surffix + '*<<*', False):
                # pre_status = st.session_state['menu_option']
                st.session_state['curr_page'] = st.session_state.get('curr_page', 0) - 1
                st.session_state['pre_page'] = st.session_state.get('curr_page', 0) - 1
                if (st.session_state.get('curr_page') == 1) | (st.session_state['pre_page']==0):
                    disable = False
                elif st.session_state.get('curr_page') == 0:
                    disable = True

            st.button('*<<*', key = key_surffix + '*<<*', disabled = disable)
        return st.session_state.get('curr_page', 0)
    def GetCurrentpage(self):
        return self.curr_page


lat1 = math.radians(current_location['latitude'])
lon1 = math.radians(current_location['longitude'])


# @st.cache_data
def haversine(lat2, lon2):
    # Chuyển đổi độ sang radian
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Kích thước bán kính của Trái Đất
    R = 6371.0  # Đơn vị: kilometer
    
    # Tính độ chênh lệch giữa các vĩ độ và kinh độ
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Áp dụng công thức haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Tính khoảng cách
    distance = R * c
    
    return distance




# @st.cache_data
def RecmtByUserLocation(df_location, from_index, to_index):
    distance_dict = {}
    df_location = df_location.copy()

    # Calculate the distance between user to each restaurant
    for id in df_location.index:
        lat2 = df_location.loc[id,'lat']
        lon2 = df_location.loc[id,'long']

        distance_dict[id] = haversine(lat2, lon2)

    recmt_df = GetTopRecmt(distance_dict, df_location, from_index, to_index, False)
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
            st.markdown('**👨‍💻Owner**: Trần Quốc Toàn')
            st.markdown('**🏠Place**:  Go Vap, Ho Chi Minh City')
            st.markdown('**📞Phone**: +(84)942 557 560')
            st.markdown('**✉️Email**: quoctoan11102000@gmail.com')
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

height_size = 3
width_size = 7

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
    # st.markdown(" - Tình hình nhà hàng quán ăn ở TPHCM trên nền tảng Foody.")
    st.markdown(" - Chi tiết về dự án.")
    


# 5. Add on_change callback
def on_change(key):
    selection = st.session_state[key]
    # st.write(f"Selection changed to {selection}")
    
menu = option_menu(None, ["Gợi ý món ăn", "Chi tiết dự án"],
                        icons=['house', 'cloud-upload'],
                        on_change=on_change, key='menu', orientation="horizontal")
"---"





if menu == "Gợi ý món ăn":
    try:
        st.subheader("**Các quán ăn gần đây**", divider ='rainbow')
        ShowGridOfItem(height_size= defaul_height
                    , width_size= width
                    , data= RecmtByUserLocation(data, 0, defaul_height * width * 10)
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
    # st.write(data)


    
    ShowGridOfItem(height_size = main_height, width_size = width, data = data, key_surffix = "main")


    # "---"

if menu == "Chi tiết dự án":
    st.title("How this App works?")
    st.header("Flow Chart:")
    # Create a graphlib graph object
    content_col = st.columns([1,2])
    with content_col[0]:
        graph = graphviz.Digraph()
        graph.edge('Data Collection', 'Data Processing', 'Data Explodatory')
        graph.edge('Data Processing', 'Building Model to Cluster')
        graph.edge('Building Model to Cluster', 'Deploy to Streamlit app')

        st.graphviz_chart(graph, use_container_width=False)
    with content_col[1]:
        st.write("The idea is building a Recommendation system base on real data, I chose a fictional topic: building the best restaurant/food store suggestion system for users.")
        st.subheader("Step I: Data Collection")
        st.write("- Pharse 1: Crawling Restaurant/Food store in HCM city from ShopeeFood (https://shopeefood.vn/ho-chi-minh/food/deals):")
        with st.expander("View sample data after crawling"):
            st.write(pd.read_csv("tests.csv").head(10).iloc[:,1:])
        
        st.write("- Pharse 2: Crawling each Restaurant/Food store's info from Foody:")
        with st.expander("View sample data after crawling"):
            st.write(pd.read_csv("RestaurantInfo.csv").head(10))
    with st.expander("Function to crawling Data"):
        st.code('''
def CrawlFoodData():
    all_web_item = driver.find_elements(By.CSS_SELECTOR,".item-restaurant .item-content")
    result = pd.DataFrame(columns=['name_res', 'img_link', 'food_link','favorite_tag', 'is_quality_mer','address','promotion'])
    for i in all_web_item:
        try: 
            ##get name
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.info-restaurant .name-res')))
            name_res = i.find_element(By.CSS_SELECTOR, '.info-restaurant .name-res').text
            ##get img link
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.img-restaurant img')))
            link_img = i.find_element(By.CSS_SELECTOR, '.img-restaurant img').get_attribute("src")
            ##get url link
            food_link = i.get_attribute("href")
            ##
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.img-restaurant')))
            favorite_tag = i.find_element(By.CSS_SELECTOR, '.img-restaurant').text
            ##get tag
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.info-restaurant .icon.icon-quality-merchant')))
            try:
                is_quality_mer = i.find_element(By.CSS_SELECTOR, '.info-restaurant .icon.icon-quality-merchant').get_attribute('title')
            except NoSuchElementException:
                is_quality_mer = np.nan
            ##get address
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.info-restaurant .address-res')))
            address_res = i.find_element(By.CSS_SELECTOR, '.info-restaurant .address-res').text
            ##get promotion
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.info-restaurant .content-promotion')))
            promotion = i.find_element(By.CSS_SELECTOR, '.info-restaurant .content-promotion').text
        except:
            print("Lỗi ở item: ", all_web_item.index(i+1))
        
        record = pd.DataFrame([[name_res, link_img, food_link, favorite_tag, is_quality_mer, address_res, promotion]]
                              ,columns=['name_res', 'img_link', 'food_link','favorite_tag', 'is_quality_mer','address','promotion'])
        result = pd.concat([result, record])
    return result
                ''', language = 'python')
    st.write("- Techniques: Selenium, Multi Threading, Python programing")
    st.subheader("Step II: Data Processing")
    st.write("Transform collected data to the right format for analysis")        
    st.write("- Remove the ouliners and record with the incorrect format")     
    st.write("- Convert the numerical columne to the right format")
    st.write("- Create the new feature for each restaurant/food store")
    st.write("- Explodatory Analysis about the distribution and the quality of the dataset")
    st.write("- Techniques: Python programing, Data processing, Data Visullization")
    st.subheader("Step III: Build Model to cluster")
    st.write("Apply Unsupervised Learning to cluster all the record of dataset")
    st.write("Techniques: K-means Clustering, Bag of Words, Countvectorize,...")
    st.subheader("Step IV: Deploy recommendation system")
    st.write("Disploy the recommendation system base on:")
    st.write("- Similarity Food Store/Restaurant: Recommend base on the cluster of Step III")
    st.write("- User Location: Calculation distance base on coordinate of restaurant and user location")
    st.write("- Other features: Number of visit, Flash Sale promotion,...")
    "---"
    end = st.columns([5.5,3,5.5])
    with end[1]:
        st.subheader("THANKS FOR VISIT!")

with st.container():
    footer = Footer(
        'https://scontent.fsgn8-4.fna.fbcdn.net/v/t1.6435-9/148275772_1342911596063325_3059342905885115862_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=be3454&_nc_eui2=AeFxW95d_7EAKV5GOVyKmrTSKR1B5NW0V5gpHUHk1bRXmHza6EBMkV21mu7hTcgTQdKT1QTl1LH_pj4URfYAODsd&_nc_ohc=UiTLjMMEtdoAX_8b7lf&_nc_ht=scontent.fsgn8-4.fna&oh=00_AfB5Hewj1fFwzTcpg24zj20OIVtP75u73w3rf_Pcpb3ISQ&oe=6602AEF8',
        'https://www.linkedin.com/in/toan-tran-555a5621b/',
        'https://github.com/ToanToan110/',
        'https://www.facebook.com/profile.php?id=100010334923606'
    )
    footer.show_page()
