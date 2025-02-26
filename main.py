import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.font_manager as fm

font_path = "./data/ipaexg.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

print(font_prop.get_name())

df_Pre_Mun = pd.read_csv('./data/Pre_Mun.csv')
pre = df_Pre_Mun['Pre'].unique()

df_muni_2020 = pd.read_csv('./data/data_muni_2020.csv', index_col=0)
df_muni_2015 = pd.read_csv('./data/data_muni_2015.csv', index_col=0)
df_muni_2010 = pd.read_csv('./data/data_muni_2010.csv', index_col=0)
df_score_2020 = pd.read_csv('./data/sdgs_score_2020.csv', index_col=0)
df_score_2015 = pd.read_csv('./data/sdgs_score_2015.csv', index_col=0)
df_score_2010 = pd.read_csv('./data/sdgs_score_2010.csv', index_col=0)
df_goal_vars = pd.read_csv('./data/variable_each.csv', index_col=0)

def select_df(year:int, type:str):
    if type == 'muni':
        if year == 2020:
            df_muni = df_muni_2020
        elif year == 2015:
            df_muni = df_muni_2015
        else:
            df_muni = df_muni_2010
        return df_muni
    else:
        if year == 2020:
            df_score = df_score_2020
        elif year == 2015:
            df_score = df_score_2015
        else:
            df_score = df_score_2010
        return df_score

def cal_draw_factor_analysis(df_goal_vars:pd.DataFrame, df_muni:pd.DataFrame, muni:list, goal_num:int, year:int):
    var = ast.literal_eval(df_goal_vars.iloc[goal_num-1, 0])
    factor = df_muni[var].loc[muni].values - df_muni[var].mean().values
    score = []
    for i in factor:
        score.append(i.sum())
    
    if len(factor) == 1:
        df_factor_analysis = pd.DataFrame(data=factor.reshape(-1,1))
        df_factor_analysis = df_factor_analysis.transpose()
    else:
        df_factor_analysis = pd.DataFrame(data=factor)
    df_factor_analysis.columns = var
    df_factor_analysis.index = df_muni.loc[muni].index.values
    
    print(df_factor_analysis)

    fig, ax = plt.subplots()  # fig, ax を作成
    df_factor_analysis.plot.bar(stacked=True, ax=ax, legend=True, cmap='RdYlBu')  # axを指定して描画
    ax.scatter(x=muni, y=score, marker='D', color='k', s=40, label='スコア')
    ax.scatter(x=muni, y=score, marker='D', color='w', s=20)
    ax.axhline(0, color='r', linestyle='dashed', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    ax.set_title(f'Goal_{goal_num} Factor Analysis in {year}')

    st.pyplot(fig) 

def draw_score_change(df_score_2020:pd.DataFrame, df_score_2015:pd.DataFrame, df_score_2010:pd.DataFrame, muni:list, goal_num:str):
    df_score = pd.DataFrame([df_score_2020.loc[muni, [goal_num]].transpose().values.tolist()[0], 
                       df_score_2015.loc[muni, [goal_num]].transpose().values.tolist()[0], 
                       df_score_2010.loc[muni, [goal_num]].transpose().values.tolist()[0]], 
                      columns=muni, 
                      index=[2020, 2015, 2010])
    fig, ax = plt.subplots()
    df_score.plot(ax=ax)
    ax.set_xticks([2010, 2015, 2020])
    ax.set_ylabel('Score')
    ax.set_title(f'Goal_{goal_num} score change')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    st.pyplot(fig)

def draw_sdgs_mapping(df_score:pd.DataFrame, muni:str, year:int, back:bool = False):
    if muni not in df_score.index:
        return print(f'"{muni}" does not exist.')
    
    x = np.array(range(1, 18))
    y = df_score[df_score.index == muni].values[0]
    x_ave = np.array(range(1, 18))
    y_ave = np.array([100]*17)
    
    width = np.array([1]*len(x))/(len(x))*2*np.pi
    theta = np.insert(width.cumsum(), 0, 0)[:-1]
    width_ave = np.array([1]*len(x_ave))/(len(x_ave))*2*np.pi
    theta_ave = np.insert(width_ave.cumsum(), 0, 0)[:-1]
    
    color = ['#e5243b', '#dda63a', '#4c9f38', '#c5192d', '#ff3a21', '#26bde2', '#fcc30b', '#a21942', '#fd6925', '#dd1367', '#fd9d24', '#bf8b2e', '#3f7e44', '#0a97d9', '#56c02b', '#00589d', '#19486a']

    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'polar': True})  # fig, ax を作成（極座標）

    if back:
        ax.bar(theta_ave, y_ave, width=width_ave, align='edge', color=color, alpha=0.1)
    ax.bar(theta_ave, y_ave/1.99, width=width_ave, align='edge', color=['r']*17, alpha=0.7, label='mean')
    ax.bar(theta_ave, y_ave/2.01, width=width_ave, align='edge', color=['w']*17, alpha=1)
    if back:
        ax.bar(theta_ave, y_ave/2.025, width=width_ave, align='edge', color=color, alpha=0.1)
    ax.bar(theta, y, width=width, align='edge', color=color, alpha=1, label='SDGs Score')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360/17), labels=['']*17)
    ax.set_rgrids(np.arange(0, 110, 25), angle=0)
    ax.spines['polar'].set_visible(False)
    ax.set_title(f'SDGs_Score {muni} in {year}', loc='left')
    
    st.pyplot(fig)

st.title('JYR-Index Analysis Board')

# Sidebarの選択肢を定義する
options = ["Factor Analysis", "Look SDGs Score"]
choice = st.sidebar.selectbox("Select an Mode", options)

# Mainコンテンツの表示を変える
if choice == "Factor Analysis":
    st.header("# Factor Analysis Mode")

    FA_mode = st.selectbox(
        'Chose "Factor Analysis Mode"', 
        ['one_year', 'multi_year']
    )
    
    if FA_mode == 'one_year':
        Prefecture = st.selectbox(
            'Chose "Prefecture"', 
            pre
        )

        Municipalities = st.multiselect(
            'Chose "Municipalities"', 
            df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values, 
            default= df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values[0]
        )

        SDGs_Goal = st.multiselect(
            'Chose "SDGs Goal"',
            list(range(1, 18)), 
            default= [1]
        )

        dataset_year = st.selectbox(
            'Chose "dataset year"',
            [2020, 2015, 2010]
        )

        if st.button('Apply'):
            for i in range(0, len(SDGs_Goal)):
                cal_draw_factor_analysis(df_goal_vars, select_df(dataset_year, 'muni'), Municipalities, SDGs_Goal[i], dataset_year)
    else:
        Prefecture = st.selectbox(
            'Chose "Prefecture"', 
            pre
        )
        
        Municipalities = st.multiselect(
            'Chose "Municipalities"', 
            df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values, 
            default= df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values[0]
        )

        SDGs_Goal = st.multiselect(
            'Chose "SDGs Goal"',
            list(range(1, 18)), 
            default= [1]
        )
        
        if st.button('Apply'):
            for i in range(0, len(SDGs_Goal)):
                col1, col2 = st.columns(2)
                with col1:
                    draw_score_change(df_score_2010, df_score_2015, df_score_2020, Municipalities, str(SDGs_Goal[i]))
                    st.subheader(f'Goal {SDGs_Goal[i]}')
                for j in range(2010, 2025, 5):
                    with col2:
                        cal_draw_factor_analysis(df_goal_vars, select_df(j, 'muni'), Municipalities, SDGs_Goal[i], j)

    
elif choice == "Look SDGs Score":
    st.header("# Look SDGs Score")
    st.text(font_path, font_prop.get_name())
    Prefecture = st.selectbox(
        'Chose "Prefecture"', 
        pre
    )

    Municipalities = st.multiselect(
        'Chose "Municipalities"', 
        df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values, 
        default= df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values[0]
    )

    dataset_year = st.multiselect(
        'Chose "dataset year"',
        [2020, 2015, 2010], 
        default= [2020]
    )
    
    if st.button('Apply'):
        for i in range(0, len(Municipalities)):
            st.subheader(f'{Municipalities[i]}')
            if len(dataset_year) == 2:
                col1, col2 = st.columns(2)
                with col1:
                    draw_sdgs_mapping(select_df(dataset_year[0], 'score'), Municipalities[i], dataset_year[0])
                with col2:
                    draw_sdgs_mapping(select_df(dataset_year[1], 'score'), Municipalities[i], dataset_year[1])
            elif len(dataset_year) == 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    draw_sdgs_mapping(select_df(dataset_year[0], 'score'), Municipalities[i], dataset_year[0])
                with col2:
                    draw_sdgs_mapping(select_df(dataset_year[1], 'score'), Municipalities[i], dataset_year[1])
                with col3:
                    draw_sdgs_mapping(select_df(dataset_year[2], 'score'), Municipalities[i], dataset_year[2])
            else:
                draw_sdgs_mapping(select_df(dataset_year[0], 'score'), Municipalities[i], dataset_year[0])

