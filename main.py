import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.font_manager as fm
import os
from PIL import Image

fpath = "./data/NotoSansJP-Regular.otf"
prop = fm.FontProperties(fname=fpath)

df_Pre_Mun = pd.read_csv('./data/Pre_Mun.csv')
pre = df_Pre_Mun['Pre'].unique()

designated_muni = ['札幌市', '仙台市', 'さいたま市', '千葉市', '横浜市', '川崎市', '相模原市', '新潟市', '静岡市', '浜松市', '名古屋市', '京都市', '大阪市', '堺市', '神戸市', '岡山市', '広島市', '北九州市', '福岡市', '熊本市']

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
    ax.set_xticklabels(muni, fontproperties=prop)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', prop=prop)
    ax.set_title(f'Goal_{goal_num} Factor Analysis in {year}')

    st.pyplot(fig) 

def draw_score_change(df_score_2020:pd.DataFrame, df_score_2015:pd.DataFrame, df_score_2010:pd.DataFrame, muni:list, goal_num:str):
    df_score = pd.DataFrame([df_score_2010.loc[muni, [goal_num]].transpose().values.tolist()[0], 
                       df_score_2015.loc[muni, [goal_num]].transpose().values.tolist()[0], 
                       df_score_2020.loc[muni, [goal_num]].transpose().values.tolist()[0]], 
                      columns=muni, 
                      index=[2020, 2015, 2010])
    fig, ax = plt.subplots()
    df_score.plot(ax=ax)
    ax.set_xticks([2010, 2015, 2020])
    ax.set_ylabel('Score')
    ax.set_title(f'Goal_{goal_num} score change')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop=prop)
    
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
    ax.set_title(f'SDGs_Score {muni} in {year}', loc='left', fontproperties=prop)
    
    st.pyplot(fig)

def draw_compare_score(df_score:pd.DataFrame, mun:str, df_target:pd.Series, target_name:str):
    color = ['#e5243b', '#dda63a', '#4c9f38', '#c5192d', '#ff3a21', '#26bde2', '#fcc30b', '#a21942', '#fd6925', '#dd1367', '#fd9d24', '#bf8b2e', '#3f7e44', '#0a97d9', '#56c02b', '#00589d', '#19486a']
    
    fig, ax = plt.subplots()
    ax.bar(x=list(range(1, 18)), height=df_target, width=1, alpha=0.5, color=color, label=target_name)
    ax.bar(x=list(range(1, 18)), height=df_score.loc[mun], color=color, label=mun)
    ax.set_xticks(ticks=list(range(1, 18)))
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop=prop)
    
    st.pyplot(fig)

st.title('JYR-Index Analysis Board')

# Sidebarの選択肢を定義する
options = ["Factor Analysis", "Comparing Mode", "Look SDGs Score"]
choice = st.sidebar.selectbox("Select an Mode", options)

# Mainコンテンツの表示を変える
if choice == "Factor Analysis":
    st.header("# Factor Analysis Mode")
    
    try:
        col_pre, col_mun = st.columns(2)
        with col_pre:
            Prefecture = st.multiselect(
                'Chose "Prefecture"', 
                pre, 
                default= pre[0]
            )
        
        pre_val = []
        for pre_ in Prefecture:
            pre_val += df_Pre_Mun[df_Pre_Mun['Pre']==pre_]['Mun'].values.tolist()
        
        with col_mun:
            Municipalities = st.multiselect(
                'Chose "Municipalities"', 
                pre_val, 
                default= pre_val[0]
            )

        SDGs_Goal = st.multiselect(
            'Chose "SDGs Goal"',
            list(range(1, 18)), 
            default= [1]
        )
        
        if st.button('Apply'):
            for i in range(0, len(SDGs_Goal)):
                st.subheader(f'Goal {SDGs_Goal[i]}')
                col1, col2 = st.columns(2)
                with col1:
                    draw_score_change(df_score_2010, df_score_2015, df_score_2020, Municipalities, str(SDGs_Goal[i]))
                for j in range(2010, 2025, 5):
                    with col2:
                        cal_draw_factor_analysis(df_goal_vars, select_df(j, 'muni'), Municipalities, SDGs_Goal[i], j)
    except:
        st.text('エラーが発生しました。\nヒント：全ての項目でオブジェクトを選択してください。')

elif choice == "Comparing Mode":
    st.header("# Comparing Mode")

    col_pre, col_mun = st.columns(2)
    with col_pre:
        Prefecture = st.selectbox(
            'Chose "Prefecture"', 
            pre
        )
    
    with col_mun:
        Municipalities = st.selectbox(
            'Chose "Municipalities"', 
            df_Pre_Mun[df_Pre_Mun['Pre']==Prefecture]['Mun'].values
        )
    
    col_target_pre, col_target_mun = st.columns(2)
    with col_target_pre:
        Target_Prefecture = st.selectbox(
            'Chose "Target Prefecture"', 
            ['その他'] + pre.tolist()
        )
    
    if Target_Prefecture == 'その他':
        mun = ['目標値', '政令指定都市']
    else:
        mun = df_Pre_Mun[df_Pre_Mun['Pre']==Target_Prefecture]['Mun'].values
    
    with col_target_mun:
        Target_Municipalities = st.selectbox(
            'Chose "Target Municipalities"', 
            mun, 
        )
    
    if Target_Municipalities == '目標値':
        muni_list = [Municipalities]
        compare_target = []
        for goal in range(0, 17):
            compare_target.append(float(df_score_2020.iloc[:, goal].sort_values(ascending=False).iloc[:5].mean()))
        compare_target_name = '目標値'
    elif Target_Municipalities == '政令指定都市':
        muni_list = [Municipalities]
        compare_target = df_score_2020.loc[['札幌市', '仙台市', 'さいたま市', '千葉市', '横浜市', '川崎市', '相模原市', '新潟市', '静岡市', '浜松市', '名古屋市', '京都市', '大阪市', '堺市', '神戸市', '岡山市', '広島市', '北九州市', '福岡市', '熊本市']].mean()
        compare_target_name = '政令指定都市'
    else:
        if Municipalities == Target_Municipalities:
            muni_list = [Municipalities]
            compare_target = [50]*17
            compare_target_name = '全国平均'
        else:
            muni_list = [Municipalities, Target_Municipalities]
            compare_target = df_score_2020.loc[Target_Municipalities]
            compare_target_name = Target_Municipalities
        
    
    if st.button('Apply'):
        st.subheader('比較グラフ', divider='gray')
        draw_compare_score(df_score_2020, Municipalities, compare_target, compare_target_name)
    
        st.subheader(f'総合スコア（{round(df_score_2020.mean(axis=1)[Municipalities], 2)}）', divider='gray')
        col_gross_1, col_gross_2, col_gross_3, col_gross_4= st.columns(4)
        with col_gross_1:
            draw_sdgs_mapping(df_score_2020, Municipalities, 2020)
        with col_gross_2:
            draw_sdgs_mapping(df_score_2015, Municipalities, 2015)
        with col_gross_3:
            draw_sdgs_mapping(df_score_2010, Municipalities, 2010)
        with col_gross_4:
            if Municipalities in designated_muni:
                st.write('全国（政令指定都市）順位：')
                st.write(int(df_score_2020.mean(axis=1).rank(ascending=False, method='min')[Municipalities]),'（', int(df_score_2020.loc[designated_muni].mean(axis=1).rank(ascending=False, method='min')[Municipalities]), '）位')
            else:
                st.write('全国順位：')
                st.write(int(df_score_2020.mean(axis=1).rank(ascending=False, method='min')[Municipalities]), '位')
            st.write('変化率：')
            st.write(int(df_score_2020.mean(axis=1).rank(ascending=False, method='min')[Municipalities])/int(df_score_2010.mean(axis=1).rank(ascending=False, method='min')[Municipalities]), '％')
        
        st.subheader(f'最高スコア（{round(df_score_2020.loc[Municipalities].max(), 2)}）', divider='gray')
        col_high_1, col_high_2, col_high_3 = st.columns([1, 1.5, 1.5])
        with col_high_1:
            img = Image.open(f'./data/Goals/sdg_icon_{int(df_score_2020.loc[Municipalities].idxmax()):02}_ja-768x768.png')
            st.image(img)
        with col_high_2:
            if Municipalities in designated_muni:
                st.write('全国（政令指定都市）順位：')
                st.write(int(df_score_2020.iloc[:, int(df_score_2020.loc[Municipalities].idxmax())-1].rank(ascending=False, method='min')[Municipalities]), '（', int(df_score_2020.loc[designated_muni].iloc[:, int(df_score_2020.loc[Municipalities].idxmax())-1].rank(ascending=False, method='min')[Municipalities]),  '）位')
            else:
                st.write('全国順位：')
                st.write(int(df_score_2020.iloc[:, int(df_score_2020.loc[Municipalities].idxmax())-1].rank(ascending=False, method='min')[Municipalities]), '位')
        with col_high_3:
            st.write('変化率：')
            st.write(int(df_score_2020.iloc[:, int(df_score_2020.loc[Municipalities].idxmax())-1].rank(ascending=False, method='min')[Municipalities])/int(df_score_2010.iloc[:, int(df_score_2020.loc[Municipalities].idxmax())-1].rank(ascending=False, method='min')[Municipalities]), '％')
        col1, col2 = st.columns(2)
        with col2:
            cal_draw_factor_analysis(df_goal_vars, df_muni_2020, muni_list, int(df_score_2020.loc[Municipalities].idxmax()), 2020)
        with col1:
            draw_score_change(df_score_2020, df_score_2015, df_score_2010, muni_list, str(df_score_2020.loc[Municipalities].idxmax()))
        
        
        st.subheader(f'最低スコア（{round(df_score_2020.loc[Municipalities].min(), 2)}）', divider='gray')
        col_low_1, col_low_2, col_low_3 = st.columns([1, 1.5, 1.5])
        with col_low_1:
            img = Image.open(f'./data/Goals/sdg_icon_{int(df_score_2020.loc[Municipalities].idxmin()):02}_ja-768x768.png')
            st.image(img)
        with col_low_2:
            if Municipalities in designated_muni:
                st.write('全国（政令指定都市）順位')
                st.write(int(df_score_2020.iloc[:, int(df_score_2020.loc[Municipalities].idxmin())-1].rank(ascending=False, method='min')[Municipalities]), '（', int(df_score_2020.loc[designated_muni].iloc[:, int(df_score_2020.loc[Municipalities].idxmin())-1].rank(ascending=False, method='min')[Municipalities]), '）位')
            else:
                st.write('全国順位：')
                st.write(int(df_score_2020.iloc[:, int(df_score_2020.loc[Municipalities].idxmin())-1].rank(ascending=False, method='min')[Municipalities]), '位')
        with col_low_3:
            st.write('変化率：')
            st.write(int(df_score_2020.iloc[:, int(df_score_2020.loc[Municipalities].idxmin())-1].rank(ascending=False, method='min')[Municipalities])/int(df_score_2010.iloc[:, int(df_score_2020.loc[Municipalities].idxmin())-1].rank(ascending=False, method='min')[Municipalities]), '％')
        col1, col2 = st.columns(2)
        with col2:
            cal_draw_factor_analysis(df_goal_vars, df_muni_2020, muni_list, int(df_score_2020.loc[Municipalities].idxmin()), 2020)
        with col1:
            draw_score_change(df_score_2020, df_score_2015, df_score_2010, muni_list, str(df_score_2020.loc[Municipalities].idxmin()))



elif choice == "Look SDGs Score":
    st.header("# Look SDGs Score")
    try:
        col_pre, col_mun = st.columns(2)
        with col_pre:
            Prefecture = st.multiselect(
                'Chose "Prefecture"', 
                pre, 
                default= pre[0]
            )
        
        pre_val = []
        for pre_ in Prefecture:
            pre_val += df_Pre_Mun[df_Pre_Mun['Pre']==pre_]['Mun'].values.tolist()
        
        with col_mun:
            Municipalities = st.multiselect(
                'Chose "Municipalities"', 
                pre_val, 
                default= pre_val[0]
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
    except:
        st.text('エラーが発生しました。\nヒント：全ての項目でオブジェクトを選択してください。')