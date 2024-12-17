import pandas as pd
from st_aggrid import AgGrid
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
import altair as alt
import io
import os


ACTUAL_TEST_PATH = "./actual_test.csv"
ANSEWR_FILE_PATH = "./PBL05((工数予測))_演習03_解答.csv"

# ---- ---- ---- ----

DEV_TYPE = "開発モード"
NOMAL_TYPE = "通常モード"

# ---- ---- ---- ----

RADIO_EVAL_NONE = "評価値/残差 表示しない"
RADIO_EVAL_TABLE = "評価値/残差 データ表(Streamlit標準 表形式)"
RADIO_EVAL_DF =  "評価値/残差 データ表(Streamlit標準 DataFrame形式)", 

RADIO_DETAIL_NONE = "詳細データ 表示しない"
RADIO_DETAIL_TABLE = "詳細データ(Streamlit標準 表形式)"
RADIO_DETAIL_DF = "詳細データ(Streamlit標準 DataFrame形式)"
RADIO_DETAIL_AGGRID = "詳細データ(AgGrid)"

RADIO_GRAPH_NONE = "グラフ 表示しない"
RADIO_MATPLOTLIB = "残差グラフ(matplotlib)"
RADIO_ALTAIR = "残差グラフ(Altair)"
RADIO_PLOTLY = "残差グラフ(Plotly)"
RADIO_BOKEH = "残差グラフ(Bokeh)"
RADIO_PYGWALKER = "PyGWalker"

# ---- ---- ---- ----

def calc_score(y : pd.Series, y_hat : pd.Series, name : str):
    # MAEの計算
    mae = mean_absolute_error(y, y_hat)
    # RMSEの計算
    rmse = np.sqrt(mean_squared_error(y, y_hat))

    # RMSLEの計算
    # yとy_hatに負の値が含まれないことを確認
    if (y < 0).any() or (y_hat < 0).any():
        st.error(f"RMSLEは負の値を含むデータには適用できません({name})")
        rmsle = np.nan
    else:
        rmsle = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(y_hat)))

    return mae, rmse, rmsle

def calculate_statistics(df: pd.DataFrame, column_name: str) -> pd.Series:
    statistics = df[column_name].describe() # 基本統計量
    statistics['var'] = df[column_name].var() # 分散

    return statistics.to_dict()

def score_series_helper(statistics, mae, rmse, rmsle, name):
    statistics["mae"] = mae
    statistics["rmse"] = rmse
    statistics["rmsle"] = rmsle
    df_score = pd.Series(data=statistics, name=name)

    df_score_rename = df_score.rename(
            {
                "mae": "評価値 MAE", "rmse": "評価値 RMSE(参考)", "rmsle": "評価値 RMSLE(参考)",
                "mean": "残差 平均値", "std": "残差 標準偏差", "var": "残差 分散",
                "min": "残差 最小値", "max": "残差 最大値",
                "25%": "残差 第1四分位数(25%)", "50%": "残差 中央値(第2四分位数,50%)", "75%": "残差 第3四分位数(75%)", 
            }
        )
    df_score_drop = df_score_rename.drop("count")

    df_score_reindex = df_score_drop.reindex(index = [
                "評価値 MAE", "評価値 RMSE(参考)", "評価値 RMSLE(参考)",
                "残差 平均値", "残差 標準偏差", "残差 分散",
                "残差 最小値", "残差 第1四分位数(25%)", "残差 中央値(第2四分位数,50%)", "残差 第3四分位数(75%)", "残差 最大値",
    ])

    return df_score_reindex

def calc_score_detail(df : pd.DataFrame, name : str):
    mae_nwt, rmse_nwt, rmsle_nwt = calc_score(df["正味作業時間(正解)"], df["正味作業時間(予測)"], f"{name} 正味作業時間")
    statistics_nwt= calculate_statistics(df, "正味作業時間(残差)")
    df_data_nwt = score_series_helper(statistics_nwt, mae_nwt, rmse_nwt, rmsle_nwt, name=f"正味作業時間({name})")

    mae_awt, rmse_awt, rmsle_awt = calc_score(df["付帯作業時間(正解)"], df["付帯作業時間(予測)"], f"{name} 付帯作業時間")
    statistics_awt= calculate_statistics(df, "付帯作業時間(残差)")
    df_data_awt = score_series_helper(statistics_awt, mae_awt, rmse_awt, rmsle_awt, name=f"付帯作業時間({name})")

    return df_data_nwt, df_data_awt

# ---- ---- ---- ----

def matplotlib_helper_sub(title: str, y_pred, y_true):
    # 散布図を作成
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, y_true, c='blue', s=64, label=f'Predicted vs Actual {title}', alpha=0.7)

    # データの範囲を計算
    min_value = min(min(y_pred), min(y_true))
    max_value = max(max(y_pred), max(y_true))

    # Y = X の直線を作成
    plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Y = X')

    # グラフのタイトルと軸ラベルを設定
    plt.title(f'Predicted vs Actual Scatter Plot {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 軸の範囲を設定
    plt.xlim([min_value, max_value])
    plt.ylim([min_value, max_value])

    # 凡例を追加
    plt.legend()

    # グリッドを追加
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return plt

def sns_hist_helper(df, column_name: str, title: str):
    plt.figure(figsize=(7, 6))
    sns.histplot(df[column_name], kde=True, bins=10)
    plt.title(f"Distribution of Residual with KDE {title} ", fontsize=14)
    plt.xlabel(f"Residual {title} ", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    return plt

def matplotlib_helper(df, title: str):
    # 正味作業時間のプロット
    plt_nwt = matplotlib_helper_sub(f"({title} : net working time)", df["正味作業時間(予測)"], df["正味作業時間(正解)"])
    st.pyplot(plt_nwt)
    plt_nwt.close()

    plt_hist_nwt = sns_hist_helper(df, "正味作業時間(残差)", f"({title} : net working time)")
    st.pyplot(plt_hist_nwt)
    plt_hist_nwt.close()

    # 付帯作業時間のプロット
    plt_awt = matplotlib_helper_sub(f"({title} : ancillary work time)", df["付帯作業時間(予測)"], df["付帯作業時間(正解)"])
    st.pyplot(plt_awt)
    plt_awt.close()

    plt_hist_awt = sns_hist_helper(df, "付帯作業時間(残差)", f"({title} : ancillary working time)")
    st.pyplot(plt_hist_awt)
    plt_hist_awt.close()

# ---- ---- ---- ----

def altair_helper_sub(title: str, y_pred, y_true):
    # データフレームを作成
    data = pd.DataFrame({
        'Predicted': y_pred,
        'Actual': y_true
    })

    # データの範囲を計算
    min_value = min(min(y_pred), min(y_true))
    max_value = max(max(y_pred), max(y_true))

    # 散布図を作成
    scatter = alt.Chart(data).mark_circle(size=60, opacity=0.7, color='blue').encode(
        x=alt.X('Predicted', scale=alt.Scale(domain=[min_value, max_value])),
        y=alt.Y('Actual', scale=alt.Scale(domain=[min_value, max_value])),
        tooltip=['Predicted', 'Actual']
    ).properties(
        title=f'Predicted vs Actual Scatter Plot {title}',
        width=700,
        height=600
    )

    # Y = X の直線を作成
    line_data = pd.DataFrame({
        'x': [min_value, max_value],
        'y': [min_value, max_value]
    })
    line = alt.Chart(line_data).mark_line(color='red', strokeDash=[5, 5]).encode(
        x='x',
        y='y'
    )

    return scatter + line

def altair_hist_helper(df, column_name, title):
    # データを抽出
    data = df[[column_name]].dropna()
    
    # ヒストグラムを作成
    hist = alt.Chart(data).mark_bar(opacity=0.5).encode(
        alt.X(column_name, bin=alt.Bin(maxbins=30), title='残差'),
        alt.Y('count()', title='頻度')
    )
    
    # KDEを作成
    kde = alt.Chart(data).transform_density(
        column_name,
        as_=[column_name, 'density'],
        extent=[data[column_name].min(), data[column_name].max()]
    ).mark_line().encode(
        alt.X(column_name, title='残差'),
        alt.Y('density:Q', title='KDE')
    )
    
    # ヒストグラムとKDEを重ねて表示し、二次軸を追加
    chart = alt.layer(hist, kde).resolve_scale(
        y='independent'
    ).properties(
        title=f"残差 ヒストグラム & KDE ({title})"
    )

    return chart

def altair_helper(df, title: str):
    st.altair_chart(altair_helper_sub(f"({title} : 正味作業時間)", df["正味作業時間(予測)"], df["正味作業時間(正解)"]))
    st.altair_chart(altair_hist_helper(df, "正味作業時間(残差)", f"{title} : 正味作業時間"))

    st.altair_chart(altair_helper_sub(f"({title} : 付帯作業時間)", df["付帯作業時間(予測)"], df["付帯作業時間(正解)"]))
    st.altair_chart(altair_hist_helper(df, "付帯作業時間(残差)", f"{title} : 付帯作業時間"))

# ---- ---- ---- ----

def plotly_helper_sub(title: str, y_pred, y_true):
    # 散布図を作成
    scatter = go.Scatter(
        x=y_pred, 
        y=y_true, 
        mode='markers',
        name=f'Predicted vs Actual {title}',
        marker=dict(color='blue', size=8)
    )

    # データの範囲を計算
    min_value = min(min(y_pred), min(y_true))
    max_value = max(max(y_pred), max(y_true))

    # Y = X の直線を作成
    line = go.Scatter(
        x=[min_value, max_value], 
        y=[min_value, max_value], 
        mode='lines',
        name='Y = X',
        line=dict(color='red', dash='dash')
    )

    # 図を作成
    fig = go.Figure()
    fig.add_trace(scatter)
    fig.add_trace(line)

    # レイアウト設定
    fig.update_layout(
        title=f'Predicted vs Actual Scatter Plot {title}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        showlegend=True,
        xaxis=dict(range=[min_value, max_value]),
        yaxis=dict(range=[min_value, max_value]),
        width=700,
        height=600
    )

    return fig

def plotly_hist_helper(df, column_name: str, title: str):
    # 指定された列からデータを抽出
    data = df[column_name].dropna().values
    # ヒストグラムとKDEを含むdistplotを作成
    fig = ff.create_distplot([data], group_labels=[column_name], bin_size=0.2)

    # レイアウトを更新してタイトルを追加
    fig.update_layout(
        title=f"残差 ヒストグラム & KDE ({title})",
        xaxis_title="残差",
        yaxis_title="頻度"
    )

    return fig

def plotly_helper(df, title: str):
    st.plotly_chart(plotly_helper_sub(f"({title} : 正味作業時間)", df["正味作業時間(予測)"], df["正味作業時間(正解)"]))
    st.plotly_chart(plotly_hist_helper(df, "正味作業時間(残差)", f"{title} : 正味作業時間"))

    st.plotly_chart(plotly_helper_sub(f"({title} : 付帯作業時間)", df["付帯作業時間(予測)"], df["付帯作業時間(正解)"]))
    st.plotly_chart(plotly_hist_helper(df, "付帯作業時間(残差)", f"{title} : 付帯作業時間"))

# ---- ---- ---- ----

def bokeh_helper_sub(title: str, y_pred, y_true):
    # 散布図を作成
    p = figure(
        title=f'Predicted vs Actual Scatter Plot {title}', 
        x_axis_label='Predicted', 
        y_axis_label='Actual', 
        width=700, 
        height=600
    )

    # 散布図の追加
    p.circle(y_pred, y_true, size=8, color="blue", alpha=0.7, legend_label=f'Predicted vs Actual {title}')

    # データの範囲を計算
    min_value = min(min(y_pred), min(y_true))
    max_value = max(max(y_pred), max(y_true))

    # Y = X の直線を追加
    p.line([min_value, max_value], [min_value, max_value], line_dash="dashed", line_color="red", legend_label="Y = X")

    # 凡例の位置を設定
    p.legend.location = "top_left"

    return p

def bokeh_hist_helper(df, column_name: str, title: str):
    # データを抽出
    data = df[column_name].dropna().values
    
    # ヒストグラムを作成
    hist, edges = np.histogram(data, bins=30, density=True)
    
    # Bokehプロットを作成
    p_hist = figure(title=f"残差 ヒストグラム ({title})", x_axis_label='残差', y_axis_label='頻度', width=800, height=400)
    
    # ヒストグラムを追加
    p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.5, line_color="black")

    return p_hist

def bokeh_helper(df, title: str):
    # 正味作業時間のプロット
    fig_nwt = bokeh_helper_sub(f"({title} : 正味作業時間)", df["正味作業時間(予測)"], df["正味作業時間(正解)"])
    st.bokeh_chart(fig_nwt)

    st.bokeh_chart(bokeh_hist_helper(df, "正味作業時間(残差)", f"{title} : 正味作業時間"))

    # 付帯作業時間のプロット
    fig_awt = bokeh_helper_sub(f"({title} : 付帯作業時間)", df["付帯作業時間(予測)"], df["付帯作業時間(正解)"])
    st.bokeh_chart(fig_awt)

    st.bokeh_chart(bokeh_hist_helper(df, "付帯作業時間(残差)", f"{title} : 付帯作業時間"))

# ---- ---- ---- ----

def app(dev_mode):
    data_expander= st.expander("データ登録", expanded=True)

    with data_expander:
        if not dev_mode:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("作業実績データ")
                actual_test_file = st.file_uploader("Competitionサイトで公開された作業実績データファイル(actual_test.csv)をアップロード", type="csv")

            with col2:
                st.subheader("正解データ")
                answer_file = st.file_uploader("Competitionサイトで公開された正解データファイル(PBL05((工数予測))_演習03_解答.csv)をアップロード", type="csv")

            with col3:
                st.subheader("投稿データ")
                submit_file = st.file_uploader("投稿ファイル(csv)をアップロード", type="csv")
        else:
            st.subheader("投稿データ")
            submit_file = st.file_uploader("投稿ファイル(csv)をアップロード", type="csv")

    if dev_mode:
        actual_test_df = pd.read_csv(ACTUAL_TEST_PATH)

        with open(ANSEWR_FILE_PATH, "rb") as answer_file:
            answer_file_str = answer_file.read()
        answer_file_lines = answer_file_str.splitlines()
        answer_file_str = b"\n".join(answer_file_lines[1:])
        io_str = io.BytesIO(answer_file_str)
        anser_df = pd.read_csv(io_str, header=None)
        anser_df.columns = ["index", "正味作業時間(正解)", "付帯作業時間(正解)"]
    else:
        if actual_test_file is not None:
            actual_test_df = pd.read_csv(actual_test_file)

        if answer_file is not None:
            answer_line = answer_file.getvalue().splitlines()
            answer_data = b"\n".join(answer_line[1:])
            answer_buff = io.BytesIO(answer_data)

            anser_df = pd.read_csv(answer_buff, header=None)
            anser_df.columns = ["index", "正味作業時間(正解)", "付帯作業時間(正解)"]

    if submit_file is not None:
        submit_df = pd.read_csv(submit_file, header=None)
        submit_df.columns = ["index", "正味作業時間(予測)", "付帯作業時間(予測)"]

    calc_enable = False

    if not dev_mode:
        if actual_test_file is not None and answer_file is not None and submit_file is not None:
            calc_enable = True
    else:
        if submit_file is not None:
            calc_enable = True

    if calc_enable:
        # st.dataframe(actual_test_df)

        # ---- VVV 計算処理 VVV ----

        work_df = pd.merge(actual_test_df[["index", "号機名"]], anser_df)
        merged_df = pd.merge(work_df, submit_df)

        merged_df["正味作業時間(残差)"] = merged_df["正味作業時間(正解)"] - merged_df["正味作業時間(予測)"]
        merged_df["付帯作業時間(残差)"] = merged_df["付帯作業時間(正解)"] - merged_df["付帯作業時間(予測)"]

        combined_series_ans = pd.concat([merged_df["正味作業時間(正解)"], merged_df["付帯作業時間(正解)"]], axis=0)
        combined_series_prd = pd.concat([merged_df["正味作業時間(予測)"], merged_df["付帯作業時間(予測)"]], axis=0)
        combined_series_err = pd.concat([merged_df["正味作業時間(残差)"], merged_df["付帯作業時間(残差)"]], axis=0)
        conbined_df = pd.DataFrame({ "正解": combined_series_ans, "予測": combined_series_prd, "残差": combined_series_err})

        gluer_df = merged_df[merged_df["号機名"] == "グルアー"]
        printer_df = merged_df[merged_df["号機名"] != "グルアー"]
        p2_df = merged_df[merged_df["号機名"] == "2号機"]
        p4_df = merged_df[merged_df["号機名"] == "4号機"]
        p6_df = merged_df[merged_df["号機名"] == "6号機"]
        p7_df = merged_df[merged_df["号機名"] == "7号機"]
        p8_df = merged_df[merged_df["号機名"] == "8号機"]

        mae_all, rmse_all, rmsle_all = calc_score(conbined_df["正解"], conbined_df["予測"], "全体")
        statistics_all= calculate_statistics(conbined_df, "残差")
        score_all = score_series_helper(statistics_all, mae_all, rmse_all, rmsle_all, name="全体")

        mae_nwt, rmse_nwt, rmsle_nwt = calc_score(merged_df["正味作業時間(正解)"], merged_df["正味作業時間(予測)"], "正味作業時間")
        statistics_nwt= calculate_statistics(merged_df, "正味作業時間(残差)")
        score_nwt = score_series_helper(statistics_nwt, mae_nwt, rmse_nwt, rmsle_nwt, name="正味作業時間")

        mae_awt, rmse_awt, rmsle_awt = calc_score(merged_df["付帯作業時間(正解)"], merged_df["付帯作業時間(予測)"], "付帯作業時間")
        statistics_awt= calculate_statistics(merged_df, "付帯作業時間(残差)")
        score_awt = score_series_helper(statistics_awt, mae_awt, rmse_awt, rmsle_awt, name="付帯作業時間")

        # gluer_df
        score_nwt_g, score_awt_g = calc_score_detail(gluer_df, "グルアー")

        # printer_df
        score_nwt_p, score_awt_p = calc_score_detail(printer_df, "印刷機")

        # p2_df
        score_nwt_p2, score_awt_p2 = calc_score_detail(p2_df, "2号機")
        # p4_df
        score_nwt_p4, score_awt_p4 = calc_score_detail(p4_df, "4号機")
        # p6_df
        score_nwt_p6, score_awt_p6 = calc_score_detail(p6_df, "6号機")
        # p7_df
        score_nwt_p7, score_awt_p7 = calc_score_detail(p7_df, "7号機")
        # p8_df
        score_nwt_p8, score_awt_p8 = calc_score_detail(p8_df, "8号機")

        score_concat_df = pd.concat([
            score_all,
            score_nwt,
            score_awt,
            score_nwt_g,
            score_awt_g,
            score_nwt_p,
            score_awt_p,
        ], axis=1)

        score_concat_nwt_p_df = pd.concat([
            score_nwt_p2,
            score_nwt_p4,
            score_nwt_p6,
            score_nwt_p7,
            score_nwt_p8,
        ], axis=1)

        score_concat_awt_p_df = pd.concat([
            score_awt_p2,
            score_awt_p4,
            score_awt_p6,
            score_awt_p7,
            score_awt_p8,
        ], axis=1)

        # ---- AAA 計算処理 AAA ----

        dowunload_csv = "PBL05 演習03 評価値/残差 計算\n"
        dowunload_csv += "評価値/残差 データ\n"
        dowunload_csv += score_concat_df.to_csv()
        dowunload_csv += "\n"
        dowunload_csv += "評価値/残差 データ 印刷機 詳細\n"
        dowunload_csv += score_concat_nwt_p_df.to_csv()
        dowunload_csv += "\n"
        dowunload_csv += score_concat_awt_p_df.to_csv()
        dowunload_csv += "\n詳細データ\n"
        dowunload_csv += merged_df.to_csv(index=None)

        st.divider()

        socre_all_mae = score_all["評価値 MAE"]
        formatted_mae = f"{socre_all_mae:.4f}".replace(".", "_")

        st.download_button(
            label="CSVで計算したデータをダウンロード",
            data=dowunload_csv,
            file_name=f'MAE_{formatted_mae}_PBL05_exercise03_score_{submit_file.name}.csv',
            mime='text/csv',
        )

        eval_expander = st.expander("評価値/残差統計情報", expanded=False)
        detail_expander = st.expander("残差詳細情報", expanded=False)
        graph_expander = st.expander("グラフ", expanded=False)

        with eval_expander:
            select_eval_mode = st.radio(
                label="評価値/残差統計情報 モード選択",
                options=[
                    RADIO_EVAL_NONE,
                    RADIO_EVAL_TABLE,
                    RADIO_EVAL_DF,
                ],
                horizontal=True,
                label_visibility="hidden"
            )

            if select_eval_mode == RADIO_EVAL_TABLE:
                st.header("評価値/残差 データ表(Streamlit標準 表形式)")

                st.subheader("評価値/残差 データ")

                st.table(score_concat_df)

                st.divider()

                st.subheader("評価値/残差 データ 印刷機 詳細")

                st.table(score_concat_nwt_p_df)
                st.table(score_concat_awt_p_df)

            if select_eval_mode == RADIO_EVAL_DF:
                st.header("評価値/残差 データ表(Streamlit標準 DataFrame形式)")

                st.subheader("評価値/残差 データ")

                st.dataframe(score_concat_df)

                st.divider()

                st.subheader("評価値/残差 データ 印刷機 詳細")

                st.dataframe(score_concat_nwt_p_df)
                st.dataframe(score_concat_awt_p_df)

        with detail_expander:
            select_detail_mode = st.radio(
                label="残差詳細情報 モード選択",
                options=[
                    RADIO_DETAIL_NONE,
                    RADIO_DETAIL_TABLE,
                    RADIO_DETAIL_DF,
                    RADIO_DETAIL_AGGRID,
                ],
                horizontal=True,
                label_visibility="hidden"
            )

            if select_detail_mode == RADIO_DETAIL_TABLE:
                st.header("詳細データ(Streamlit標準 表形式)")
                st.table(merged_df)

            if select_detail_mode == RADIO_DETAIL_DF:
                st.header("詳細データ(Streamlit標準 DataFrame形式)")
                st.dataframe(merged_df)

            if select_detail_mode == RADIO_DETAIL_AGGRID:
                st.header("詳細データ(AgGrid)")
                AgGrid(merged_df, fit_columns_on_grid_load=True)

        with graph_expander:
            select_graph_mode = st.radio(
                label="モード選択",
                options=[
                    RADIO_GRAPH_NONE,
                    RADIO_MATPLOTLIB,
                    RADIO_ALTAIR,
                    RADIO_PLOTLY,
                    RADIO_BOKEH,
                    RADIO_PYGWALKER,
                ],
                horizontal=True,
                label_visibility="hidden"
            )

            if select_graph_mode == RADIO_MATPLOTLIB:
                st.header("残差グラフ(matplotlib)")

                plt_all = matplotlib_helper_sub("(All)", conbined_df["予測"], conbined_df["正解"])
                st.pyplot(plt_all)
                plt_all.close()

                plt_hist_all = sns_hist_helper(conbined_df, "残差", "(All)")
                st.pyplot(plt_hist_all)
                plt_hist_all.close()

                st.divider()

                matplotlib_helper(gluer_df, "Gluer")
                matplotlib_helper(printer_df, "Printer")

                st.divider()

                matplotlib_helper(p2_df, "Printer No.2")
                matplotlib_helper(p4_df, "Printer No.4")
                matplotlib_helper(p6_df, "Printer No.6")
                matplotlib_helper(p7_df, "Printer No.7")
                matplotlib_helper(p8_df, "Printer No.8")

            if select_graph_mode == RADIO_ALTAIR:
                st.header("残差グラフ(Altair)")

                plt_all = altair_helper_sub("(全体)", conbined_df["予測"], conbined_df["正解"])
                st.altair_chart(plt_all)

                st.altair_chart(altair_hist_helper(conbined_df, "残差", "全体"))

                st.divider()

                altair_helper(gluer_df, "グルアー")
                altair_helper(printer_df, "印刷機")

                st.divider()

                altair_helper(p2_df, "2号機")
                altair_helper(p4_df, "4号機")
                altair_helper(p6_df, "6号機")
                altair_helper(p7_df, "7号機")
                altair_helper(p8_df, "8号機")

            if select_graph_mode == RADIO_PLOTLY:
                st.header("残差グラフ(Plotly)")

                plt_all = plotly_helper_sub("(全体)", conbined_df["予測"], conbined_df["正解"])
                st.plotly_chart(plt_all)

                st.plotly_chart(plotly_hist_helper(conbined_df, "残差", "全体"))

                st.divider()

                plotly_helper(gluer_df, "グルアー")
                plotly_helper(printer_df, "印刷機")

                st.divider()

                plotly_helper(p2_df, "2号機")
                plotly_helper(p4_df, "4号機")
                plotly_helper(p6_df, "6号機")
                plotly_helper(p7_df, "7号機")
                plotly_helper(p8_df, "8号機")

            if select_graph_mode == RADIO_BOKEH:
                st.header("残差グラフ(Bokeh)")

                plt_all = bokeh_helper_sub("(全体)", conbined_df["予測"], conbined_df["正解"])
                st.bokeh_chart(plt_all)

                st.bokeh_chart(bokeh_hist_helper(conbined_df, "残差", "全体"))

                st.divider()

                bokeh_helper(gluer_df, "グルアー")
                bokeh_helper(printer_df, "印刷機")

                st.divider()

                bokeh_helper(p2_df, "2号機")
                bokeh_helper(p4_df, "4号機")
                bokeh_helper(p6_df, "6号機")
                bokeh_helper(p7_df, "7号機")
                bokeh_helper(p8_df, "8号機")

            if select_graph_mode == RADIO_PYGWALKER:
                st.header("PyGWalker")

                pyg_app = StreamlitRenderer(merged_df)
                pyg_app.explorer()

# ---- ---- ---- ----

if __name__ == "__main__":
    st.set_page_config(
        page_title="MDXQ2024 PBL05 演習03 評価値/残差 計算アプリ",
        layout="wide"
    )
    st.title("MDXQ2024 PBL05 演習03 評価値/残差 計算アプリ")

    dev_mode_enable = False
    if os.path.isfile(ACTUAL_TEST_PATH):
        if os.path.isfile(ANSEWR_FILE_PATH):
            dev_mode_enable = True

    fix_dev_mode = False
    # fix_dev_mode = True

    dev_mode = False
    if dev_mode_enable:
        if not fix_dev_mode:
            select_dev_mode = st.radio(
                    label="モード選択",
                    options=[DEV_TYPE, NOMAL_TYPE],
                    horizontal=True,
                    label_visibility="hidden"
                )
            if select_dev_mode == DEV_TYPE:
                dev_mode = True
        else:
            dev_mode = True


    app(dev_mode)
