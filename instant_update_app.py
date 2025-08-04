import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import math
import plotly.graph_objects as go

# ページ設定
st.set_page_config(layout="wide", page_title="クラウチングスタート姿勢分析")

# セッション状態の初期化
def initialize_session_state():
    if "keypoints" not in st.session_state:
        st.session_state.keypoints = {}
    if "selected_joint" not in st.session_state:
        st.session_state.selected_joint = None
    if "click_data" not in st.session_state:
        st.session_state.click_data = None

initialize_session_state()

st.title("🏃 クラウチングスタート姿勢分析（クリック選択版）")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    mode = st.selectbox("分析モード", ["セット姿勢", "飛び出し分析"])
    show_feedback = st.checkbox("フィードバック表示", value=True)
    joint_size = st.slider("関節点サイズ", 5, 20, 10)
    line_width = st.slider("線の太さ", 1, 8, 3)
    st.divider()
    st.header("🔧 調整方法")
    adjustment_mode = st.selectbox(
        "調整モード",
        ["❶ クリック選択", "❷ プルダウン選択", "❸ 方向キー調整", "❹ 一括表示"]
    )
    if st.button("🔄 AI検出をやり直す"):
        keys_to_delete = ["keypoints", "selected_joint", "click_data"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ... (既存の計算関数群は同じなので省略) ...

def create_clickable_plot(img, keypoints, joint_size, line_width, img_width, img_height):
    """クリック可能なプロット作成"""
    import io
    import base64
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    fig = go.Figure()
    
    # 背景画像
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{img_str}",
            xref="x", yref="y",
            x=0, y=img_height,
            sizex=img_width, sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首",
        "C7": "⑨ 第7頸椎"
    }
    
    # 線を描画
    lines = [
        ("LShoulder", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
        ("RShoulder", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
        ("LShoulder", "RShoulder"), ("LHip", "RHip"),
    ]
    
    for a, b in lines:
        if a in keypoints and b in keypoints:
            x1, y1 = keypoints[a]
            x2, y2 = keypoints[b]
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[img_height - y1, img_height - y2],
                mode='lines',
                line=dict(color='red', width=line_width),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 関節点を描画
    for name, (x, y) in keypoints.items():
        color = 'lightgreen' if name == st.session_state.selected_joint else 'yellow'
        size = joint_size * 2.5 if name == st.session_state.selected_joint else joint_size * 2
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[img_height - y],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(color='red', width=3)
            ),
            text=name.replace('Shoulder', '肩').replace('Hip', '股').replace('Knee', '膝').replace('Ankle', '足'),
            textposition="top center",
            hovertext=f"{joint_names_jp.get(name, name)}<br>クリックで選択",
            hoverinfo='text',
            name=name,
            customdata=[name]
        ))
    
    fig.update_layout(
        xaxis=dict(range=[0, img_width], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, img_height], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=min(700, img_height + 100),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def quick_adjustment_controls(selected_joint, keypoints, img_width, img_height):
    """クイック調整コントロール"""
    if not selected_joint or selected_joint not in keypoints:
        st.info("👆 画像上の関節点をクリックして選択してください")
        return
    
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首",
        "C7": "⑨ 第7頸椎"
    }
    
    current_x, current_y = keypoints[selected_joint]
    
    st.success(f"🎯 選択中: **{joint_names_jp.get(selected_joint, selected_joint)}**")
    
    # 方向キー風調整
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        col_up1, col_up2, col_up3 = st.columns([1, 1, 1])
        with col_up2:
            if st.button("🔼", key="up", help="上に移動"):
                new_y = max(0, current_y - 5)
                st.session_state.keypoints[selected_joint] = (current_x, new_y)
                st.rerun()
        
        col_mid1, col_mid2, col_mid3 = st.columns([1, 1, 1])
        with col_mid1:
            if st.button("◀️", key="left", help="左に移動"):
                new_x = max(0, current_x - 5)
                st.session_state.keypoints[selected_joint] = (new_x, current_y)
                st.rerun()
        with col_mid2:
            st.metric("現在位置", f"({current_x}, {current_y})")
        with col_mid3:
            if st.button("▶️", key="right", help="右に移動"):
                new_x = min(img_width, current_x + 5)
                st.session_state.keypoints[selected_joint] = (new_x, current_y)
                st.rerun()
        
        col_down1, col_down2, col_down3 = st.columns([1, 1, 1])
        with col_down2:
            if st.button("🔽", key="down", help="下に移動"):
                new_y = min(img_height, current_y + 5)
                st.session_state.keypoints[selected_joint] = (current_x, new_y)
                st.rerun()
    
    # 精密調整
    st.divider()
    st.write("**📐 精密調整**")
    col_x, col_y = st.columns(2)
    
    with col_x:
        new_x = st.number_input(
            "X座標", 
            min_value=0, max_value=img_width, 
            value=current_x, step=1,
            key=f"{selected_joint}_precise_x"
        )
    
    with col_y:
        new_y = st.number_input(
            "Y座標", 
            min_value=0, max_value=img_height, 
            value=current_y, step=1,
            key=f"{selected_joint}_precise_y"
        )
    
    if (new_x, new_y) != (current_x, current_y):
        st.session_state.keypoints[selected_joint] = (new_x, new_y)
        st.rerun()

# ... (その他の既存関数も同様に省略) ...

# メイン処理
uploaded_file = st.file_uploader("📷 画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # ... (画像読み込み、AI姿勢推定は既存と同じ) ...
        
        # レイアウト
        col_image, col_inputs = st.columns([2, 1])
        
        with col_image:
            if adjustment_mode == "❶ クリック選択":
                st.subheader("🎯 関節点をクリックして選択")
                st.info("💡 関節点をクリック → 右側で微調整")
                
                fig = create_clickable_plot(img, st.session_state.keypoints, joint_size, line_width, w, h)
                plot_data = st.plotly_chart(fig, use_container_width=True, key="clickable_plot")
                
                # クリックイベントのシミュレーション（制限あり）
                st.write("**🖱️ 関節点リスト（クリックで選択）**")
                joint_names_jp = {
                    "LShoulder": "① 左肩", "RShoulder": "② 右肩",
                    "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
                    "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
                    "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首",
                    "C7": "⑨ 第7頸椎"
                }
                
                cols = st.columns(3)
                joint_list = list(st.session_state.keypoints.keys())
                for i, joint in enumerate(joint_list):
                    with cols[i % 3]:
                        if st.button(
                            joint_names_jp.get(joint, joint),
                            key=f"select_{joint}",
                            type="primary" if joint == st.session_state.selected_joint else "secondary"
                        ):
                            st.session_state.selected_joint = joint
                            st.rerun()
            
            else:
                # その他の表示方法
                pass
        
        with col_inputs:
            if adjustment_mode == "❶ クリック選択":
                quick_adjustment_controls(st.session_state.selected_joint, st.session_state.keypoints, w, h)
            # ... (分析処理等は既存と同じ) ...

    except Exception as e:
        st.error(f"🚨 エラーが発生しました: {str(e)}")

else:
    st.info("📷 クラウチングスタートの画像をアップロードしてください。")
    st.markdown("""
    ### 🎯 実用的な代替案
    
    **❶ クリック選択**: 関節点ボタンをクリック → 方向キーで移動
    **❷ プルダウン選択**: 従来通りの選択方式
    **❸ 方向キー調整**: ゲーム感覚で微調整
    **❹ 一括表示**: 全関節点を同時表示
    
    ### 💡 推奨使用方法
    1. AI自動検出で大まかな位置を取得
    2. クリック選択で関節点を選ぶ
    3. 方向キー（🔼▶️🔽◀️）で素早く調整
    4. 精密調整で最終微調整
    """)
