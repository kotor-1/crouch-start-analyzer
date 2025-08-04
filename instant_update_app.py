import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import math
import os

# --- ページ設定 ---
st.set_page_config(layout="wide")
st.title("🏃 クラウチングスタート姿勢分析（共有・修正版）")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("設定")
    show_feedback = st.checkbox("フィードバック表示", value=True)
    show_com = st.checkbox("重心線表示", value=True)
    joint_size = st.slider("関節点サイズ", 6, 20, 10)
    st.divider()
    
    # 手動調整セクション
    st.header("🔧 手動調整")
    adjustment_mode = st.selectbox(
        "調整方法",
        ["プルダウン選択", "画像下部に横並び表示"]
    )
    
    if st.button("🔄 AI検出をやり直す"):
        if "keypoints" in st.session_state:
            del st.session_state["keypoints"]
        st.rerun()

# --- MediaPipeとヘルパー関数 ---
mp_pose = mp.solutions.pose

@st.cache_resource
def load_model():
    """MediaPipeモデルをキャッシュしてロード"""
    return mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3
    )

def get_font(font_filename="NotoSansJP-Regular.ttf", size=16):
    """フォントファイルを読み込む。見つからなければデフォルトを使用"""
    try:
        # スクリプトと同じディレクトリにあるフォントファイルを試す
        return ImageFont.truetype(font_filename, size)
    except IOError:
        # 見つからなければデフォルトフォントを使用
        st.warning(f"フォントファイル '{font_filename}' が見つかりません。デフォルトフォントで表示します。")
        return ImageFont.load_default()

def calculate_angle(p1, p2, p3):
    """3点から角度を計算する"""
    try:
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in [p1, p2, p3]):
            return None
        a, b, c = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(p3, dtype=float)
        ab, cb = a - b, c - b
        ab_norm, cb_norm = np.linalg.norm(ab), np.linalg.norm(cb)
        if ab_norm == 0 or cb_norm == 0: return None
        cosine = np.clip(np.dot(ab, cb) / (ab_norm * cb_norm), -1.0, 1.0)
        return round(np.degrees(np.arccos(cosine)), 1)
    except:
        return None

def calculate_hip_ground_angle(hip_pos, knee_pos):
    """股関節から膝への線と地面の角度を計算する"""
    try:
        dx = knee_pos[0] - hip_pos[0]
        dy = knee_pos[1] - hip_pos[1]
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        return round(180 - angle_deg if angle_deg > 90 else angle_deg, 1)
    except:
        return None

def evaluate_angles(front_angle, rear_angle, front_hip_angle):
    """角度を評価し、フィードバックと色を返す"""
    feedback, colors = [], ["info", "info", "info"]
    if front_angle is not None:
        colors[0] = "success" if 80 <= front_angle <= 100 else "error"
        if colors[0] == "error": feedback.append(f"前足の膝角度 {front_angle:.1f}° → 90°に近づけましょう。")
    if rear_angle is not None:
        colors[1] = "success" if 120 <= rear_angle <= 135 else "error"
        if colors[1] == "error": feedback.append(f"後足の膝角度 {rear_angle:.1f}° → 適正範囲(120-135°)を意識しましょう。")
    if front_hip_angle is not None:
        colors[2] = "success" if 40 <= front_hip_angle <= 60 else "error"
        if colors[2] == "error": feedback.append(f"前足股関節角度 {front_hip_angle:.1f}° → 適正範囲(40-60°)を意識しましょう。")
    return feedback, colors

def calculate_com(points):
    """重心線（COM）を概算する"""
    try:
        upper_com = np.mean([points[p] for p in ["LShoulder", "RShoulder", "LHip", "RHip"]], axis=0)
        lower_com = np.mean([points[p] for p in ["LAnkle", "RAnkle"]], axis=0)
        return tuple(map(float, upper_com)), tuple(map(float, lower_com))
    except:
        return None, None

def draw_pose_on_image(img, keypoints, joint_size, show_com):
    """画像に姿勢を描画する"""
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    font = get_font(size=16)
    
    joint_numbers = {"LShoulder":"1","RShoulder":"2","LHip":"3","RHip":"4","LKnee":"5","RKnee":"6","LAnkle":"7","RAnkle":"8"}
    lines = [("LShoulder","LHip"),("LHip","LKnee"),("LKnee","LAnkle"),("RShoulder","RHip"),("RHip","RKnee"),("RKnee","RAnkle"),("LShoulder","RShoulder"),("LHip","RHip")]
    
    for a, b in lines:
        if a in keypoints and b in keypoints:
            draw.line([keypoints[a], keypoints[b]], fill="red", width=3)
            
    for name, (x, y) in keypoints.items():
        if name in joint_numbers:
            radius = max(joint_size // 2, 6)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill="yellow", outline="red", width=3)
            
            number = joint_numbers[name]
            text_radius, text_x, text_y = 12, x + radius + 15, y
            draw.ellipse([text_x-text_radius, text_y-text_radius, text_x+text_radius, text_y+text_radius], fill="white", outline="black", width=2)
            
            bbox = draw.textbbox((0, 0), number, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((text_x - text_width/2, text_y - text_height/2), number, fill="black", font=font)

    if show_com:
        upper_com, lower_com = calculate_com(keypoints)
        if upper_com and lower_com:
            draw.line([upper_com, lower_com], fill="blue", width=4)
            h, w = img.height, img.width
            if abs(upper_com[1] - lower_com[1]) > 1:
                slope = (upper_com[0] - lower_com[0]) / (upper_com[1] - lower_com[1])
                x_intersect = lower_com[0] + slope * (h - lower_com[1])
                if 0 <= x_intersect <= w:
                    draw.line([lower_com, (int(x_intersect), h-1)], fill="blue", width=4)
                    draw.line([(int(x_intersect - 15), h-5), (int(x_intersect + 15), h-5)], fill="blue", width=8)
    return new_img

def manual_adjustment_ui(keypoints, img_width, img_height, mode):
    """手動調整UI"""
    joint_names_jp = {"LShoulder":"①左肩","RShoulder":"②右肩","LHip":"③左股関節","RHip":"④右股関節","LKnee":"⑤左膝","RKnee":"⑥右膝","LAnkle":"⑦左足首","RAnkle":"⑧右足首"}
    st.subheader(f"🎯 関節点の手動調整（{mode}）")

    if mode == "プルダウン選択":
        selected_joint = st.selectbox("調整する関節点を選択", options=list(joint_names_jp.keys()), format_func=lambda x: joint_names_jp[x])
        joints_to_adjust = [selected_joint]
    else: # 横並び表示
        joints_to_adjust = list(joint_names_jp.keys())
    
    cols = st.columns(4)
    for i, joint in enumerate(joints_to_adjust):
        if joint in keypoints:
            container = cols[i % 4] if mode != "プルダウン選択" else st
            with container:
                if mode != "プルダウン選択": st.write(f"**{joint_names_jp[joint]}**")
                current_x, current_y = keypoints[joint]
                new_x = st.number_input(f"横(X) {joint_names_jp[joint]}", 0, img_width, int(current_x), 1, key=f"{joint}_x", label_visibility="collapsed")
                new_y = st.number_input(f"縦(Y) {joint_names_jp[joint]}", 0, img_height, int(current_y), 1, key=f"{joint}_y", label_visibility="collapsed")
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.rerun()

# --- メインアプリ ---
if "keypoints" not in st.session_state: st.session_state.keypoints = {}

uploaded_file = st.file_uploader("📷 画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    if not st.session_state.keypoints:
        with st.spinner("🤖 AI姿勢推定中..."):
            model = load_model()
            results = model.process(img_np)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                lm_map = {name: getattr(mp_pose.PoseLandmark, f"{side}_{body_part}") for name, side, body_part in [("LShoulder","LEFT","SHOULDER"),("RShoulder","RIGHT","SHOULDER"),("LHip","LEFT","HIP"),("RHip","RIGHT","HIP"),("LKnee","LEFT","KNEE"),("RKnee","RIGHT","KNEE"),("LAnkle","LEFT","ANKLE"),("RAnkle","RIGHT","ANKLE")]}
                for name, landmark_idx in lm_map.items():
                    st.session_state.keypoints[name] = (int(lm[landmark_idx].x * w), int(lm[landmark_idx].y * h))
                st.success("✅ AI検出完了！")
            else:
                st.warning("⚠️ AI検出失敗。手動で調整してください。")
                st.session_state.keypoints = {"LShoulder":(w//4,h//4),"RShoulder":(3*w//4,h//4),"LHip":(w//4,h//2),"RHip":(3*w//4,h//2),"LKnee":(w//4,3*h//4),"RKnee":(3*w//4,3*h//4),"LAnkle":(w//4,h-50),"RAnkle":(3*w//4,h-50)}
    
    if st.session_state.keypoints:
        current_result = draw_pose_on_image(img, st.session_state.keypoints, joint_size, show_com)
        st.image(current_result, caption="現在の関節点位置", use_column_width=True)
        
        manual_adjustment_ui(st.session_state.keypoints, w, h, adjustment_mode)
        
        points = st.session_state.keypoints
        if all(j in points for j in ["LKnee", "RKnee", "LHip", "RHip", "LAnkle", "RAnkle"]):
            is_right_foot_front = points["RKnee"][0] < points["LKnee"][0]
            front_hip, front_knee, front_ankle = ("RHip", "RKnee", "RAnkle") if is_right_foot_front else ("LHip", "LKnee", "LAnkle")
            rear_hip, rear_knee, rear_ankle = ("LHip", "LKnee", "LAnkle") if is_right_foot_front else ("RHip", "RKnee", "RAnkle")

            front_angle = calculate_angle(points[front_hip], points[front_knee], points[front_ankle])
            rear_angle = calculate_angle(points[rear_hip], points[rear_knee], points[rear_ankle])
            front_hip_angle = calculate_hip_ground_angle(points[front_hip], points[front_knee])
            
            feedback, colors = evaluate_angles(front_angle, rear_angle, front_hip_angle)

            st.subheader("📊 最終分析結果")
            col1, col2, col3 = st.columns(3)
            metrics = {"前足の膝角度": front_angle, "後足の膝角度": rear_angle, "前足股関節角度": front_hip_angle}
            for i, ((label, value), color) in enumerate(zip(metrics.items(), colors)):
                with [col1, col2, col3][i]:
                    st.metric(label, f"{value:.1f}°" if value is not None else "測定不可")
                    if color == "success": st.success("✅ 理想的")
                    elif color == "error": st.error("⚠️ 要改善")
            if show_feedback and feedback:
                st.subheader("💡 改善アドバイス")
                for advice in feedback: st.info(advice)
else:
    st.info("📷 クラウチングスタートの画像をアップロードしてください。")
