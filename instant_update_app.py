import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import math

# --- ページ設定 ---
st.set_page_config(layout="wide")
st.title("🏃 クラウチングスタート姿勢分析（即座更新版）")

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

def calculate_angle(p1, p2, p3):
    """3点から角度を計算する"""
    try:
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in [p1, p2, p3]):
            return None
            
        a, b, c = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(p3, dtype=float)
        ab, cb = a - b, c - b
        
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        
        if ab_norm == 0 or cb_norm == 0:
            return None
            
        cosine = np.dot(ab, cb) / (ab_norm * cb_norm)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        
        return round(angle, 1) if not np.isnan(angle) else None
    except:
        return None

def calculate_hip_ground_angle(hip_pos, knee_pos):
    """股関節から膝への線と地面の角度を計算する"""
    try:
        # 股関節から膝へのベクトル
        dx = knee_pos[0] - hip_pos[0]
        dy = knee_pos[1] - hip_pos[1]  # Y軸は下向きが正
        
        # 地面（水平線）との角度を計算
        angle_rad = math.atan2(dy, dx)  # 下向きが正の角度
        angle_deg = math.degrees(angle_rad)
        
        # 0-90度の範囲に正規化（地面に向かう角度として）
        if angle_deg < 0:
            angle_deg = abs(angle_deg)
        elif angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return round(angle_deg, 1)
    except:
        return None

def evaluate_angles(front_angle, rear_angle, front_hip_angle):
    """角度を評価し、フィードバックと色を返す"""
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

def calculate_com(points):
    """重心線（COM）を概算する"""
    required_points = ["LShoulder", "RShoulder", "LHip", "RHip", "LAnkle", "RAnkle"]
    try:
        if all(p in points for p in required_points):
            upper_com = np.mean([
                points["LShoulder"], points["RShoulder"], 
                points["LHip"], points["RHip"]
            ], axis=0)
            lower_com = np.mean([points["LAnkle"], points["RAnkle"]], axis=0)
            return tuple(map(float, upper_com)), tuple(map(float, lower_com))
    except:
        pass
    return None, None

def draw_pose_on_image(img, keypoints, joint_size, show_com):
    """画像に姿勢を描画する（番号ラベル付き）"""
    try:
        new_img = img.copy()
        draw = ImageDraw.Draw(new_img)
        
        # 関節点の番号マッピング（修正版）
        joint_numbers = {
            "LShoulder": "1", "RShoulder": "2",
            "LHip": "3", "RHip": "4", 
            "LKnee": "5", "RKnee": "6",
            "LAnkle": "7", "RAnkle": "8"
        }
        
        # スケルトンの線を描画
        lines = [
            ("LShoulder", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
            ("RShoulder", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
            ("LShoulder", "RShoulder"), ("LHip", "RHip")
        ]
        
        for a, b in lines:
            if a in keypoints and b in keypoints:
                try:
                    x1, y1 = keypoints[a]
                    x2, y2 = keypoints[b]
                    draw.line([(int(x1), int(y1)), (int(x2), int(y2))], fill="red", width=3)
                except:
                    continue
        
        # 関節点を描画（番号付き）
        for name, (x, y) in keypoints.items():
            if name in joint_numbers:
                try:
                    radius = max(joint_size // 2, 6)
                    x, y = int(x), int(y)
                    
                    # 関節点の円を描画
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                               fill="yellow", outline="red", width=3)
                    
                    # 番号を描画（シンプルな数字）
                    number = joint_numbers[name]
                    
                    # 番号の背景円を描画（視認性向上）
                    text_radius = 12
                    text_x = x + radius + 15
                    text_y = y
                    
                    # 白い背景円
                    draw.ellipse([
                        text_x - text_radius, text_y - text_radius,
                        text_x + text_radius, text_y + text_radius
                    ], fill="white", outline="black", width=2)
                    
                    # 番号テキスト（中央揃え）
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    # テキストのサイズを取得して中央に配置
                    bbox = draw.textbbox((0, 0), number, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    draw.text((
                        text_x - text_width // 2, 
                        text_y - text_height // 2
                    ), number, fill="black", font=font)
                    
                except Exception:
                    # フォールバック：シンプルな描画
                    draw.text((x+radius+5, y-10), joint_numbers[name], fill="black")
        
        # 重心線を描画
        if show_com:
            upper_com, lower_com = calculate_com(keypoints)
            if upper_com and lower_com:
                try:
                    # 重心線（体の中心線）
                    draw.line([
                        (int(upper_com[0]), int(upper_com[1])), 
                        (int(lower_com[0]), int(lower_com[1]))
                    ], fill="blue", width=4)
                    
                    # 地面との交点を計算
                    h, w = img.height, img.width
                    if abs(upper_com[1] - lower_com[1]) > 1:
                        slope = (upper_com[0] - lower_com[0]) / (upper_com[1] - lower_com[1])
                        x_intersection = lower_com[0] + slope * (h - lower_com[1])
                        
                        if 0 <= x_intersection <= w:
                            draw.line([
                                (int(lower_com[0]), int(lower_com[1])), 
                                (int(x_intersection), h-1)
                            ], fill="blue", width=4)
                            
                            draw.line([
                                (int(x_intersection - 15), h - 5), 
                                (int(x_intersection + 15), h - 5)
                            ], fill="blue", width=8)
                except:
                    pass
        
        return new_img
    except:
        return img

def manual_adjustment_dropdown(keypoints, img_width, img_height):
    """プルダウン方式の手動調整（即座更新）"""
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首"
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
        
        # 即座に更新（値が変わったらセッション状態を更新）
        if (new_x, new_y) != (current_x, current_y):
            st.session_state.keypoints[selected_joint] = (new_x, new_y)
            st.rerun()

def manual_adjustment_horizontal(keypoints, img_width, img_height):
    """横並び方式の手動調整（即座更新）"""
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首"
    }
    
    st.subheader("🎯 関節点の手動調整（横並び表示）")
    
    # 上段（肩と股関節）
    st.write("**上半身**")
    col1, col2, col3, col4 = st.columns(4)
    upper_joints = ["LShoulder", "RShoulder", "LHip", "RHip"]
    
    for i, (col, joint) in enumerate(zip([col1, col2, col3, col4], upper_joints)):
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
                
                # 即座に更新
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.rerun()
    
    st.divider()
    
    # 下段（膝と足首）
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
                
                # 即座に更新
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.rerun()

# --- セッション状態の初期化 ---
if "keypoints" not in st.session_state:
    st.session_state.keypoints = {}

# --- メインアプリ ---
uploaded_file = st.file_uploader("📷 画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # 画像読み込み
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # 初回またはリセット時のみAI検出を実行
        if not st.session_state.keypoints:
            with st.spinner("🤖 AI姿勢推定中..."):
                model = load_model()
                if model:
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
                            except:
                                st.session_state.keypoints[name] = (w//2, h//2)
                        
                        st.success("✅ AI検出完了！下記で関節点を調整してください。")
                    else:
                        st.warning("⚠️ AI検出に失敗しました。")
                        default_positions = {
                            "LShoulder": (w//4, h//4), "RShoulder": (3*w//4, h//4),
                            "LHip": (w//4, h//2), "RHip": (3*w//4, h//2),
                            "LKnee": (w//4, 3*h//4), "RKnee": (3*w//4, 3*h//4),
                            "LAnkle": (w//4, h-50), "RAnkle": (3*w//4, h-50)
                        }
                        st.session_state.keypoints = default_positions
        
        # 現在の結果を表示
        if st.session_state.keypoints:
            current_result = draw_pose_on_image(img, st.session_state.keypoints, joint_size, show_com)
            
            st.subheader("🎯 現在の分析結果")
            st.image(current_result, caption="現在の関節点位置", use_column_width=True)
            
            # 手動調整インターフェース（画像の下に配置）
            if adjustment_mode == "プルダウン選択":
                manual_adjustment_dropdown(st.session_state.keypoints, w, h)
            else:
                manual_adjustment_horizontal(st.session_state.keypoints, w, h)
            
            # 角度計算と結果表示
            required_joints = ["LKnee", "RKnee", "LHip", "RHip", 
                             "LAnkle", "RAnkle", "LShoulder", "RShoulder"]
            
            if all(joint in st.session_state.keypoints for joint in required_joints):
                points = st.session_state.keypoints
                
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
                front_angle = calculate_angle(
                    points[front_points[0]], points[front_points[1]], points[front_points[2]]
                )
                rear_angle = calculate_angle(
                    points[rear_points[0]], points[rear_points[1]], points[rear_points[2]]
                )
                
                # 前足股関節から膝への角度（地面に対して）
                front_hip_angle = calculate_hip_ground_angle(
                    points[front_hip_points[0]], points[front_hip_points[1]]
                )

                # 評価
                feedback, colors = evaluate_angles(front_angle, rear_angle, front_hip_angle)

                # 結果表示
                st.subheader("📊 最終分析結果")
                col1, col2, col3 = st.columns(3)
                
                values = [
                    f"{front_angle:.1f}°" if front_angle else "測定不可",
                    f"{rear_angle:.1f}°" if rear_angle else "測定不可", 
                    f"{front_hip_angle:.1f}°" if front_hip_angle else "測定不可"
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
                
                # フィードバック
                if show_feedback and feedback:
                    st.subheader("💡 改善アドバイス")
                    for advice in feedback:
                        st.info(advice)

    except Exception as e:
        st.error(f"🚨 エラーが発生しました: {str(e)}")

else:
    st.info("📷 クラウチングスタートの画像をアップロードしてください。")
    
    st.markdown("""
    ### 🚀 改良点
    - **即座更新**: 数値を変更するとリアルタイムで関節点が移動
    - **縦横表示**: X座標（横方向）、Y座標（縦方向）を明確に表示
    - **角度修正**: 前足股関節から膝への地面に対する角度を計算
    - **番号表記**: 関節点の横に1,2,3...の番号を表示（文字化け解消）
    
    ### 📋 関節点番号
    - ① 左肩　② 右肩　③ 左股関節　④ 右股関節
    - ⑤ 左膝　⑥ 右膝　⑦ 左足首　⑧ 右足首
    
    ### 📐 測定角度
    - **前足の膝角度**: 股関節-膝-足首の角度
    - **後足の膝角度**: 股関節-膝-足首の角度  
    - **前足股関節角度**: 股関節から膝への線と地面の角度
    """)