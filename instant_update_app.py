import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import math
import json
import time
import os
import pandas as pd
import datetime
from typing import Dict, Tuple, List, Optional, Union, Any

# ページ設定
st.set_page_config(layout="wide", page_title="クラウチングスタート姿勢分析")

# セッション状態の初期化
def initialize_session_state():
    session_vars = {
        "keypoints": {},
        "model_loaded": False,
        "canvas_update_flag": 0,
        "last_keypoints_str": "",
        "front_angle": None,
        "rear_angle": None,
        "front_hip_angle": None,
        "lower_angle": None,
        "upper_angle": None,
        "kunoji_angle": None,
        "mode": "セット姿勢",
        "analysis_complete": False,
        "error_message": None
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

initialize_session_state()

st.title("🏃 クラウチングスタート姿勢分析 & 飛び出し分析（エラー修正完全版）")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    mode = st.selectbox("分析モード", ["セット姿勢", "飛び出し分析"], 
                         index=0 if st.session_state.mode == "セット姿勢" else 1)
    
    # モード変更を追跡
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.analysis_complete = False
    
    show_feedback = st.checkbox("フィードバック表示", value=True)
    joint_size = st.slider("関節点サイズ", 6, 20, 10)
    st.divider()
    
    st.header("🔧 手動調整")
    adjustment_mode = st.selectbox(
        "調整方法",
        ["プルダウン選択", "画像下部に横並び表示"]
    )
    
    if st.button("🔄 AI検出をやり直す"):
        keys_to_delete = ["keypoints", "model_loaded", "canvas_update_flag", 
                          "last_keypoints_str", "analysis_complete", "error_message"]
        keys_to_delete.extend(["front_angle", "rear_angle", "front_hip_angle", 
                               "lower_angle", "upper_angle", "kunoji_angle"])
        
        for key in keys_to_delete:
            if key in st.session_state:
                st.session_state[key] = None if key != "keypoints" else {}
        st.rerun()
    
    # デバッグセクション
    with st.expander("🛠️ デバッグ情報", expanded=False):
        if st.session_state.error_message:
            st.error(f"最後のエラー: {st.session_state.error_message}")
            if st.button("エラー情報をクリア"):
                st.session_state.error_message = None
                st.rerun()

# MediaPipe初期化
mp_pose = mp.solutions.pose

@st.cache_resource
def load_model():
    """MediaPipeモデルの安全な初期化"""
    try:
        model = mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3
        )
        return model
    except Exception as e:
        st.session_state.error_message = f"MediaPipeの初期化に失敗しました: {str(e)}"
        return None

def safe_calculate_angle(p1: Tuple[float, float], 
                         p2: Tuple[float, float], 
                         p3: Tuple[float, float]) -> Optional[float]:
    """安全な角度計算関数"""
    try:
        # 入力値の検証
        if not all(p is not None for p in [p1, p2, p3]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in [p1, p2, p3]):
            return None
        if not all(isinstance(coord, (int, float)) for p in [p1, p2, p3] for coord in p[:2]):
            return None
            
        # numpy配列への変換と計算
        a = np.array([float(p1[0]), float(p1[1])], dtype=np.float64)
        b = np.array([float(p2[0]), float(p2[1])], dtype=np.float64)
        c = np.array([float(p3[0]), float(p3[1])], dtype=np.float64)
        
        # ベクトル計算
        ab = a - b
        cb = c - b
        
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        
        # ゼロ除算チェック
        if ab_norm < 1e-10 or cb_norm < 1e-10:
            return None
            
        # 角度計算
        cosine = np.dot(ab, cb) / (ab_norm * cb_norm)
        cosine = np.clip(cosine, -1.0, 1.0)  # 数値誤差対策
        angle = np.degrees(np.arccos(cosine))
        
        # NaNチェック
        if np.isnan(angle) or np.isinf(angle):
            return None
            
        return round(float(angle), 1)
    except Exception as e:
        st.session_state.error_message = f"角度計算エラー: {str(e)}"
        return None

def safe_calculate_hip_ground_angle(hip_pos: Tuple[float, float], 
                                    knee_pos: Tuple[float, float]) -> Optional[float]:
    """安全な股関節角度計算"""
    try:
        # 入力値の検証
        if not all(p is not None for p in [hip_pos, knee_pos]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in [hip_pos, knee_pos]):
            return None
            
        # 座標の取得と計算
        dx = float(knee_pos[0]) - float(hip_pos[0])
        dy = float(knee_pos[1]) - float(hip_pos[1])
        
        # ゼロベクトルチェック
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return None
            
        # 角度計算
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 角度の調整
        if angle_deg < 0:
            angle_deg = abs(angle_deg)
        elif angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return round(angle_deg, 1)
    except Exception as e:
        st.session_state.error_message = f"股関節角度計算エラー: {str(e)}"
        return None

def safe_vector_angle_with_ground(p1: Tuple[float, float], 
                                  p2: Tuple[float, float]) -> Optional[float]:
    """安全なベクトル角度計算"""
    try:
        # 入力値の検証
        if not all(p is not None for p in [p1, p2]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in [p1, p2]):
            return None
            
        # ベクトル計算
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        
        v = np.array([dx, dy], dtype=np.float64)
        ground = np.array([1, 0], dtype=np.float64)
        
        # ベクトルの長さチェック
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            return None
            
        # 角度計算
        cos_theta = np.dot(v, ground) / norm_v
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数値誤差対策
        angle = np.degrees(np.arccos(cos_theta))
        
        # NaNチェック
        if np.isnan(angle) or np.isinf(angle):
            return None
            
        return round(float(angle), 1)
    except Exception as e:
        st.session_state.error_message = f"ベクトル角度計算エラー: {str(e)}"
        return None

def evaluate_angles(front_angle: Optional[float], 
                    rear_angle: Optional[float], 
                    front_hip_angle: Optional[float]) -> Tuple[List[str], List[str]]:
    """角度評価関数"""
    feedback = []
    colors = ["info", "info", "info"]
    
    if front_angle is not None:
        if abs(front_angle - 90) > 10:
            feedback.append(f"前足の膝角度 {front_angle:.1f}° → 90°に近づけましょう。")
            colors[0] = "error"
        else:
            colors[0] = "success"
    
    if rear_angle is not None:
        if rear_angle < 120 or rear_angle > 135:
            feedback.append(f"後足の膝角度 {rear_angle:.1f}° → 適正範囲(120-135°)を意識しましょう。")
            colors[1] = "error"
        else:
            colors[1] = "success"
    
    if front_hip_angle is not None:
        if front_hip_angle < 40 or front_hip_angle > 60:
            feedback.append(f"前足股関節角度 {front_hip_angle:.1f}° → 適正範囲(40-60°)を意識しましょう。")
            colors[2] = "error"
        else:
            colors[2] = "success"
    
    return feedback, colors

def evaluate_takeoff_angles(lower_angle: Optional[float], 
                            upper_angle: Optional[float], 
                            kunoji_angle: Optional[float]) -> Tuple[List[str], List[str]]:
    """飛び出し角度評価関数"""
    feedback = []
    colors = ["info", "info", "info"]
    
    if lower_angle is not None:
        if lower_angle < 30 or lower_angle > 60:
            feedback.append(f"下半身角度 {lower_angle:.1f}° → 30-60°が目安です。")
            colors[0] = "error"
        else:
            colors[0] = "success"
    
    if upper_angle is not None:
        if upper_angle < 25 or upper_angle > 55:
            feedback.append(f"上半身角度 {upper_angle:.1f}° → 25-55°が目安です。")
            colors[1] = "error"
        else:
            colors[1] = "success"
    
    if kunoji_angle is not None:
        if kunoji_angle < 150:
            feedback.append(f"くの字角度 {kunoji_angle:.1f}° → 150°以上が目安です。")
            colors[2] = "error"
        else:
            colors[2] = "success"
    
    return feedback, colors

def keypoints_to_canvas_objects(keypoints: Dict[str, Tuple[float, float]], 
                               joint_size: int) -> List[Dict[str, Any]]:
    """関節点をキャンバスオブジェクトに変換"""
    objects = []
    joint_numbers = {
        "LShoulder": "1", "RShoulder": "2", "LHip": "3", "RHip": "4",
        "LKnee": "5", "RKnee": "6", "LAnkle": "7", "RAnkle": "8", "C7": "9"
    }
    
    try:
        # 関節点の描画
        for name, (x, y) in keypoints.items():
            objects.append({
                "type": "circle",
                "left": float(x),
                "top": float(y),
                "radius": joint_size,
                "fill": "yellow",
                "stroke": "red",
                "strokeWidth": 3,
                "name": name,
                "label": joint_numbers.get(name, "")
            })
        
        # 線の描画
        lines = [
            ("LShoulder", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
            ("RShoulder", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
            ("LShoulder", "RShoulder"), ("LHip", "RHip"),
        ]
        
        for a, b in lines:
            if a in keypoints and b in keypoints:
                objects.append({
                    "type": "line",
                    "x1": float(keypoints[a][0]),
                    "y1": float(keypoints[a][1]),
                    "x2": float(keypoints[b][0]),
                    "y2": float(keypoints[b][1]),
                    "stroke": "red",
                    "strokeWidth": 3,
                })
        
        # C7から骨盤への線
        if all(k in keypoints for k in ["C7", "RHip", "LHip", "RAnkle", "LAnkle"]):
            # 前足の判定（RAnkleかLAnkleの位置で判断）
            pelvis = keypoints["RHip"] if keypoints["RAnkle"][0] > keypoints["LAnkle"][0] else keypoints["LHip"]
            objects.append({
                "type": "line",
                "x1": float(keypoints["C7"][0]),
                "y1": float(keypoints["C7"][1]),
                "x2": float(pelvis[0]),
                "y2": float(pelvis[1]),
                "stroke": "purple",
                "strokeWidth": 5,
            })
    except Exception as e:
        st.session_state.error_message = f"キャンバスオブジェクト変換エラー: {str(e)}"
    
    return objects

def extract_keypoints_from_canvas(canvas_data: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """キャンバスデータから関節点を抽出（エラー完全回避版）"""
    keypoints = {}
    
    if not canvas_data:
        return keypoints
        
    if not isinstance(canvas_data, dict) or "objects" not in canvas_data:
        return keypoints
    
    try:
        for obj in canvas_data["objects"]:
            if isinstance(obj, dict) and obj.get("type") == "circle" and "name" in obj:
                if "left" in obj and "top" in obj:
                    try:
                        name = str(obj["name"])
                        x = int(float(obj["left"]))
                        y = int(float(obj["top"]))
                        keypoints[name] = (x, y)
                    except (ValueError, TypeError):
                        continue
    except Exception as e:
        st.session_state.error_message = f"キャンバスデータ抽出エラー: {str(e)}"
    
    return keypoints

def keypoints_to_string(keypoints: Dict[str, Tuple[float, float]]) -> str:
    """関節点を文字列に変換（比較用）"""
    try:
        return json.dumps(keypoints, sort_keys=True)
    except Exception as e:
        st.session_state.error_message = f"キーポイント文字列変換エラー: {str(e)}"
        return str(keypoints)

def manual_adjustment_dropdown(keypoints: Dict[str, Tuple[float, float]], 
                              img_width: int, 
                              img_height: int) -> None:
    """プルダウン選択による手動調整"""
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首",
        "C7": "⑨ 第7頸椎"
    }
    
    st.subheader("🎯 関節点の手動調整（プルダウン選択）")
    
    selected_joint = st.selectbox(
        "調整する関節点を選択",
        options=list(joint_names_jp.keys()),
        format_func=lambda x: joint_names_jp[x]
    )
    
    if selected_joint in keypoints:
        current_x, current_y = keypoints[selected_joint]
        st.write(f"**{joint_names_jp[selected_joint]}の位置調整**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**横方向（X座標）**")
            new_x = st.number_input(
                "左右の位置", 
                min_value=0, max_value=img_width, 
                value=int(current_x),
                step=1,
                key=f"{selected_joint}_x_input",
                help="数値を変更すると即座に更新されます"
            )
        
        with col2:
            st.write("**縦方向（Y座標）**")
            new_y = st.number_input(
                "上下の位置", 
                min_value=0, max_value=img_height, 
                value=int(current_y),
                step=1,
                key=f"{selected_joint}_y_input",
                help="数値を変更すると即座に更新されます"
            )
        
        if (new_x, new_y) != (current_x, current_y):
            st.session_state.keypoints[selected_joint] = (new_x, new_y)
            st.session_state.canvas_update_flag += 1
            st.session_state.analysis_complete = False
            st.rerun()

def manual_adjustment_horizontal(keypoints: Dict[str, Tuple[float, float]], 
                                img_width: int, 
                                img_height: int) -> None:
    """横並び表示による手動調整"""
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首",
        "C7": "⑨ 第7頸椎"
    }
    
    st.subheader("🎯 関節点の手動調整（横並び表示）")
    
    # 上半身
    st.write("**上半身**")
    col1, col2, col3, col4, col5 = st.columns(5)
    upper_joints = ["LShoulder", "RShoulder", "LHip", "RHip", "C7"]
    
    for i, (col, joint) in enumerate(zip([col1, col2, col3, col4, col5], upper_joints)):
        if joint in keypoints:
            with col:
                jp_name = joint_names_jp[joint]
                current_x, current_y = keypoints[joint]
                st.write(f"**{jp_name}**")
                
                st.write("横方向(X)")
                new_x = st.number_input(
                    "X", min_value=0, max_value=img_width, 
                    value=int(current_x), step=1,
                    key=f"{joint}_x_h", label_visibility="collapsed"
                )
                
                st.write("縦方向(Y)")
                new_y = st.number_input(
                    "Y", min_value=0, max_value=img_height, 
                    value=int(current_y), step=1,
                    key=f"{joint}_y_h", label_visibility="collapsed"
                )
                
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.session_state.canvas_update_flag += 1
                    st.session_state.analysis_complete = False
                    st.rerun()
    
    st.divider()
    
    # 下半身
    st.write("**下半身**")
    col1, col2, col3, col4 = st.columns(4)
    lower_joints = ["LKnee", "RKnee", "LAnkle", "RAnkle"]
    
    for i, (col, joint) in enumerate(zip([col1, col2, col3, col4], lower_joints)):
        if joint in keypoints:
            with col:
                jp_name = joint_names_jp[joint]
                current_x, current_y = keypoints[joint]
                st.write(f"**{jp_name}**")
                
                st.write("横方向(X)")
                new_x = st.number_input(
                    "X", min_value=0, max_value=img_width, 
                    value=int(current_x), step=1,
                    key=f"{joint}_x_h2", label_visibility="collapsed"
                )
                
                st.write("縦方向(Y)")
                new_y = st.number_input(
                    "Y", min_value=0, max_value=img_height, 
                    value=int(current_y), step=1,
                    key=f"{joint}_y_h2", label_visibility="collapsed"
                )
                
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.session_state.canvas_update_flag += 1
                    st.session_state.analysis_complete = False
                    st.rerun()

def save_analysis_results() -> None:
    """現在の分析結果をCSVに保存"""
    try:
        # 分析モードに応じて必要なデータを取得
        if st.session_state.mode == "セット姿勢":
            if not all(k in st.session_state and st.session_state[k] is not None 
                      for k in ["front_angle", "rear_angle", "front_hip_angle"]):
                st.warning("分析データが不足しています。先に分析を実行してください。")
                return
            
            data = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "セット姿勢",
                "front_angle": st.session_state.front_angle,
                "rear_angle": st.session_state.rear_angle,
                "front_hip_angle": st.session_state.front_hip_angle
            }
        else:  # 飛び出し分析
            if not all(k in st.session_state and st.session_state[k] is not None 
                      for k in ["lower_angle", "upper_angle", "kunoji_angle"]):
                st.warning("分析データが不足しています。先に分析を実行してください。")
                return
            
            data = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "飛び出し分析",
                "lower_angle": st.session_state.lower_angle,
                "upper_angle": st.session_state.upper_angle,
                "kunoji_angle": st.session_state.kunoji_angle
            }
        
        # DataFrameの作成
        df = pd.DataFrame([data])
        
        # ファイルが存在するかチェックして追加または新規作成
        filename = "sprint_analysis_results.csv"
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(filename, index=False)
            except Exception as e:
                st.error(f"既存ファイルの読み込みエラー: {str(e)}")
                df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)
        
        st.success(f"分析結果を {filename} に保存しました！")
        
        # ファイルのダウンロードを許可
        try:
            with open(filename, "rb") as file:
                st.download_button(
                    label="📥 分析結果をダウンロード",
                    data=file,
                    file_name=filename,
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"ダウンロードボタン作成エラー: {str(e)}")
    
    except Exception as e:
        st.error(f"結果保存エラー: {str(e)}")
        st.session_state.error_message = f"結果保存エラー: {str(e)}"

def show_progress_chart() -> None:
    """保存された分析データから進捗グラフを表示"""
    try:
        filename = "sprint_analysis_results.csv"
        if not os.path.exists(filename):
            st.info("まだ保存されたデータがありません。分析結果を保存するとここにグラフが表示されます。")
            return
        
        df = pd.read_csv(filename)
        if len(df) < 2:
            st.info("分析データが少なすぎます。複数回の分析結果を保存するとグラフが表示されます。")
            return
        
        st.subheader("💹 経過分析")
        
        # セット姿勢と飛び出し分析のデータを分ける
        set_df = df[df["mode"] == "セット姿勢"].copy()
        takeoff_df = df[df["mode"] == "飛び出し分析"].copy()
        
        # 日付を適切にフォーマット
        for data_df in [set_df, takeoff_df]:
            if not data_df.empty:
                data_df["timestamp"] = pd.to_datetime(data_df["timestamp"])
                data_df["date"] = data_df["timestamp"].dt.strftime("%m/%d")
        
        # セット姿勢のグラフ
        if not set_df.empty and len(set_df) > 1:
            st.write("**セット姿勢の推移**")
            
            import plotly.express as px
            
            # 角度データのみを抽出
            angle_cols = ["front_angle", "rear_angle", "front_hip_angle"]
            plot_data = set_df[["date"] + [col for col in angle_cols if col in set_df.columns]].copy()
            
            # 日本語ラベルのマッピング
            angle_labels = {
                "front_angle": "前足の膝角度",
                "rear_angle": "後足の膝角度",
                "front_hip_angle": "前足股関節角度"
            }
            
            # グラフ作成
            fig = px.line(
                plot_data, x="date", 
                y=[col for col in angle_cols if col in plot_data.columns],
                title="セット姿勢の角度推移",
                labels={"value": "角度 (度)", "date": "日付", "variable": "測定項目"},
                markers=True
            )
            
            # ラベルを日本語に変更
            fig.for_each_trace(lambda t: t.update(name=angle_labels.get(t.name, t.name)))
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 飛び出し分析のグラフ
        if not takeoff_df.empty and len(takeoff_df) > 1:
            st.write("**飛び出し分析の推移**")
            
            import plotly.express as px
            
            # 角度データのみを抽出
            angle_cols = ["lower_angle", "upper_angle", "kunoji_angle"]
            plot_data = takeoff_df[["date"] + [col for col in angle_cols if col in takeoff_df.columns]].copy()
            
            # 日本語ラベルのマッピング
            angle_labels = {
                "lower_angle": "下半身角度",
                "upper_angle": "上半身角度",
                "kunoji_angle": "くの字角度"
            }
            
            # グラフ作成
            fig = px.line(
                plot_data, x="date", 
                y=[col for col in angle_cols if col in plot_data.columns],
                title="飛び出し姿勢の角度推移",
                labels={"value": "角度 (度)", "date": "日付", "variable": "測定項目"},
                markers=True
            )
            
            # ラベルを日本語に変更
            fig.for_each_trace(lambda t: t.update(name=angle_labels.get(t.name, t.name)))
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"グラフ表示エラー: {str(e)}")
        st.session_state.error_message = f"グラフ表示エラー: {str(e)}"

# メイン処理
uploaded_file = st.file_uploader("📷 画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # 画像読み込み
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_np = np.array(img)
        
        # 画像サイズの検証
        if img_np.size == 0:
            st.error("無効な画像ファイルです")
            st.stop()
            
        h, w = img_np.shape[:2]
        
        if h <= 0 or w <= 0:
            st.error("画像サイズが無効です")
            st.stop()
        
        # AI姿勢推定（初回のみ）
        if not st.session_state.keypoints:
            with st.spinner("🤖 AI姿勢推定中..."):
                model = load_model()
                if model:
                    try:
                        results = model.process(img_np)
                        if results.pose_landmarks:
                            lm = results.pose_landmarks.landmark
                            landmark_map = {
                                "LShoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
                                "RShoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                "LHip": mp_pose.PoseLandmark.LEFT_HIP,
                                "RHip": mp_pose.PoseLandmark.RIGHT_HIP,
                                "LKnee": mp_pose.PoseLandmark.LEFT_KNEE,
                                "RKnee": mp_pose.PoseLandmark.RIGHT_KNEE,
                                "LAnkle": mp_pose.PoseLandmark.LEFT_ANKLE,
                                "RAnkle": mp_pose.PoseLandmark.RIGHT_ANKLE
                            }
                            
                            for name, landmark_idx in landmark_map.items():
                                try:
                                    landmark = lm[landmark_idx]
                                    x = int(landmark.x * w)
                                    y = int(landmark.y * h)
                                    if 0 <= x <= w and 0 <= y <= h:
                                        st.session_state.keypoints[name] = (x, y)
                                except Exception:
                                    st.session_state.keypoints[name] = (w//2, h//2)
                            
                            # C7（第7頸椎）の計算
                            if "LShoulder" in st.session_state.keypoints and "RShoulder" in st.session_state.keypoints:
                                lx, ly = st.session_state.keypoints["LShoulder"]
                                rx, ry = st.session_state.keypoints["RShoulder"]
                                st.session_state.keypoints["C7"] = ((lx+rx)//2, (ly+ry)//2)
                            
                            st.success("✅ AI検出完了！下記で関節点を調整してください。")
                        else:
                            st.warning("⚠️ AI検出に失敗しました。デフォルト位置を設定します。")
                            default_positions = {
                                "LShoulder": (w//4, h//4), "RShoulder": (3*w//4, h//4),
                                "LHip": (w//4, h//2), "RHip": (3*w//4, h//2),
                                "LKnee": (w//4, 3*h//4), "RKnee": (3*w//4, 3*h//4),
                                "LAnkle": (w//4, h-50), "RAnkle": (3*w//4, h-50),
                                "C7": (w//2, h//5)
                            }
                            st.session_state.keypoints = default_positions
                    except Exception as e:
                        st.warning(f"AI姿勢推定エラー: {str(e)}。デフォルト位置を設定します。")
                        default_positions = {
                            "LShoulder": (w//4, h//4), "RShoulder": (3*w//4, h//4),
                            "LHip": (w//4, h//2), "RHip": (3*w//4, h//2),
                            "LKnee": (w//4, 3*h//4), "RKnee": (3*w//4, 3*h//4),
                            "LAnkle": (w//4, h-50), "RAnkle": (3*w//4, h-50),
                            "C7": (w//2, h//5)
                        }
                        st.session_state.keypoints = default_positions
                else:
                    st.error("MediaPipeモデルの読み込みに失敗しました")
                    st.warning("デフォルト位置を使用します")
                    default_positions = {
                        "LShoulder": (w//4, h//4), "RShoulder": (3*w//4, h//4),
                        "LHip": (w//4, h//2), "RHip": (3*w//4, h//2),
                        "LKnee": (w//4, 3*h//4), "RKnee": (3*w//4, 3*h//4),
                        "LAnkle": (w//4, h-50), "RAnkle": (3*w//4, h-50),
                        "C7": (w//2, h//5)
                    }
                    st.session_state.keypoints = default_positions

        # レイアウト
        col_image, col_inputs = st.columns([2, 1])
        
        with col_image:
            st.subheader("🎯 画像上の関節点（ドラッグで移動可）")
            
            # キャンバス描画（エラー完全回避版）
            try:
                objects = keypoints_to_canvas_objects(st.session_state.keypoints, joint_size)
                canvas_key = f"canvas_{st.session_state.canvas_update_flag}"
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    background_image=img_np,
                    update_streamlit=True,
                    height=h,
                    width=w,
                    drawing_mode="transform",
                    initial_drawing=objects,
                    key=canvas_key,
                )
                
                # キャンバス更新処理（完全エラー回避版）
                if canvas_result and canvas_result.json_data:
                    try:
                        # 文字列比較でキャンバスの変更を検出
                        current_keypoints_str = keypoints_to_string(st.session_state.keypoints)
                        canvas_keypoints = extract_keypoints_from_canvas(canvas_result.json_data)
                        
                        if canvas_keypoints:  # 空でない場合のみ比較
                            canvas_keypoints_str = keypoints_to_string(canvas_keypoints)
                            
                            # 変更があった場合のみ更新
                            if (canvas_keypoints_str != current_keypoints_str and 
                                canvas_keypoints_str != st.session_state.last_keypoints_str and
                                len(canvas_keypoints) > 0):
                                
                                # セッション状態を更新
                                st.session_state.keypoints.update(canvas_keypoints)
                                st.session_state.last_keypoints_str = canvas_keypoints_str
                                st.session_state.canvas_update_flag += 1
                                st.session_state.analysis_complete = False
                    except Exception as e:
                        st.session_state.error_message = f"キャンバス更新エラー: {str(e)}"
            except Exception as e:
                st.error(f"キャンバス描画エラー: {str(e)}")
                st.session_state.error_message = f"キャンバス描画エラー: {str(e)}"

        with col_inputs:
            # 手動調整UI
            if adjustment_mode == "プルダウン選択":
                manual_adjustment_dropdown(st.session_state.keypoints, w, h)
            else:
                manual_adjustment_horizontal(st.session_state.keypoints, w, h)
            
            # 分析処理
            try:
                points = st.session_state.keypoints
                
                if mode == "セット姿勢":
                    required_joints = ["LKnee", "RKnee", "LHip", "RHip", 
                                      "LAnkle", "RAnkle", "LShoulder", "RShoulder"]
                    
                    if all(joint in points for joint in required_joints):
                        # 前足・後足の判定
                        if points["RKnee"][0] < points["LKnee"][0]:
                            front_points = ("RHip", "RKnee", "RAnkle")
                            rear_points = ("LHip", "LKnee", "LAnkle")
                            front_hip_points = ("RHip", "RKnee")
                        else:
                            front_points = ("LHip", "LKnee", "LAnkle")
                            rear_points = ("RHip", "RKnee", "RAnkle")
                            front_hip_points = ("LHip", "LKnee")
                        
                        # 角度計算
                        front_angle = safe_calculate_angle(
                            points[front_points[0]], points[front_points[1]], points[front_points[2]]
                        )
                        rear_angle = safe_calculate_angle(
                            points[rear_points[0]], points[rear_points[1]], points[rear_points[2]]
                        )
                        front_hip_angle = safe_calculate_hip_ground_angle(
                            points[front_hip_points[0]], points[front_hip_points[1]]
                        )
                        
                        # セッション状態に保存
                        st.session_state.front_angle = front_angle
                        st.session_state.rear_angle = rear_angle
                        st.session_state.front_hip_angle = front_hip_angle
                        st.session_state.analysis_complete = True
                        
                        # 評価
                        feedback, colors = evaluate_angles(front_angle, rear_angle, front_hip_angle)
                        
                        # 結果表示
                        st.subheader("📊 最終分析結果")
                        col1, col2, col3 = st.columns(3)
                        
                        values = [
                            f"{front_angle:.1f}°" if front_angle is not None else "測定不可",
                            f"{rear_angle:.1f}°" if rear_angle is not None else "測定不可", 
                            f"{front_hip_angle:.1f}°" if front_hip_angle is not None else "測定不可"
                        ]
                        labels = ["前足の膝角度", "後足の膝角度", "前足股関節角度"]
                        
                        for i, (col, label, value, color) in enumerate(zip([col1, col2, col3], labels, values, colors)):
                            with col:
                                st.metric(label, value)
                                if color == "success":
                                    st.success("✅ 理想的")
                                elif color == "error":
                                    st.error("⚠️ 要改善")
                                else:
                                    st.info("ℹ️ 測定中")
                        
                        # フィードバック表示
                        if show_feedback and feedback:
                            st.subheader("💡 改善アドバイス")
                            for advice in feedback:
                                st.info(advice)
                                
                        # 保存ボタン
                        if st.session_state.analysis_complete:
                            if st.button("💾 この分析結果を保存"):
                                save_analysis_results()
                                show_progress_chart()
                    else:
                        st.warning("必要な関節点が不足しています")
                        
                elif mode == "飛び出し分析":
                    required_joints = ["C7", "RHip", "LHip", "RAnkle", "LAnkle"]
                    
                    if all(k in points for k in required_joints):
                        # 前足の判定
                        if points["RAnkle"][0] > points["LAnkle"][0]:
                            hip = points["RHip"]
                            ankle = points["RAnkle"]
                        else:
                            hip = points["LHip"]
                            ankle = points["LAnkle"]
                        
                        c7 = points["C7"]
                        
                        # 角度計算
                        lower_angle = safe_vector_angle_with_ground(hip, ankle)
                        upper_angle = safe_vector_angle_with_ground(c7, hip)
                        kunoji = safe_calculate_angle(c7, hip, ankle)
                        
                        # セッション状態に保存
                        st.session_state.lower_angle = lower_angle
                        st.session_state.upper_angle = upper_angle
                        st.session_state.kunoji_angle = kunoji
                        st.session_state.analysis_complete = True
                        
                        # 評価
                        feedback, colors = evaluate_takeoff_angles(lower_angle, upper_angle, kunoji)
                        
                        # 結果表示
                        st.subheader("📊 飛び出し分析結果")
                        col1, col2, col3 = st.columns(3)
                        
                        values = [
                            f"{lower_angle:.1f}°" if lower_angle is not None else "測定不可",
                            f"{upper_angle:.1f}°" if upper_angle is not None else "測定不可",
                            f"{kunoji:.1f}°" if kunoji is not None else "測定不可"
                        ]
                        labels = ["下半身角度", "上半身角度(C7-股関節)", "くの字角度(C7-股関節-足首)"]
                        
                        for i, (col, label, value, color) in enumerate(zip([col1, col2, col3], labels, values, colors)):
                            with col:
                                st.metric(label, value)
                                if color == "success":
                                    st.success("✅ 理想的")
                                elif color == "error":
                                    st.error("⚠️ 要改善")
                                else:
                                    st.info("ℹ️ 測定中")
                        
                        # フィードバック表示
                        if show_feedback and feedback:
                            st.subheader("💡 改善アドバイス")
                            for advice in feedback:
                                st.info(advice)
                                
                        # 保存ボタン
                        if st.session_state.analysis_complete:
                            if st.button("💾 この分析結果を保存"):
                                save_analysis_results()
                                show_progress_chart()
                    else:
                        st.warning("必要な関節点が不足しています")
            
            except Exception as e:
                st.error(f"分析処理エラー: {str(e)}")
                st.session_state.error_message = f"分析処理エラー: {str(e)}"
                
        # グラフ表示
        if st.session_state.analysis_complete:
            with st.expander("📈 これまでの分析データ", expanded=False):
                show_progress_chart()

    except Exception as e:
        st.error(f"🚨 エラーが発生しました: {str(e)}")
        st.session_state.error_message = f"全体処理エラー: {str(e)}"

else:
    st.info("📷 クラウチングスタートの画像をアップロードしてください。")
    st.markdown("""
    ### 🚀 主な機能と特長
    - **AI姿勢検出**: MediaPipeを使用して関節点を自動検出
    - **手動調整**: 関節点をドラッグで移動、または数値入力で微調整
    - **角度分析**: 膝・股関節の角度を計算し、理想的な姿勢と比較
    - **結果保存**: 分析結果をCSVファイルに保存して経過を追跡
    - **データ可視化**: 時間経過による姿勢の改善を確認
    - **エラー耐性**: すべての操作に対して堅牢なエラーハンドリング

    ### 📋 関節点番号
    - ① 左肩　② 右肩　③ 左股関節　④ 右股関節
    - ⑤ 左膝　⑥ 右膝　⑦ 左足首　⑧ 右足首
    - ⑨ 第7頸椎（C7）
    
    ### 💡 使い方
    1. 画像をアップロードする
    2. AIが自動検出した関節点を確認
    3. 必要に応じて関節点をドラッグまたは数値入力で調整
    4. 分析結果とアドバイスを確認
    5. 「結果を保存」ボタンでデータを記録
    6. 「これまでの分析データ」で進捗を確認
    """)
    
    # デモ機能の案内
    st.subheader("🆕 追加機能のプレビュー")
    st.markdown("""
    - **結果の保存と進捗グラフ**: 分析結果をCSVファイルに保存し、時間経過による変化をグラフで確認できます
    - **エラー耐性の強化**: すべてのステップでエラー処理を強化し、予期せぬ問題が発生しても安全に動作します
    - **型ヒントの導入**: Pythonの型ヒントにより、コードの信頼性と読みやすさが向上しました
    
    画像をアップロードして、これらの新機能をお試しください！
    """)
