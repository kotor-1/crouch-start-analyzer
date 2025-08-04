import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import math
import io
import base64

# ページ設定
st.set_page_config(layout="wide", page_title="クラウチングスタート姿勢分析")

# セッション状態の初期化
def initialize_session_state():
    if "keypoints" not in st.session_state:
        st.session_state.keypoints = {}
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

initialize_session_state()

st.title("🏃 クラウチングスタート姿勢分析 & 飛び出し分析（エラー完全解決版）")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    mode = st.selectbox("分析モード", ["セット姿勢", "飛び出し分析"])
    show_feedback = st.checkbox("フィードバック表示", value=True)
    joint_size = st.slider("関節点サイズ", 3, 15, 8)
    line_width = st.slider("線の太さ", 1, 10, 3)
    st.divider()
    st.header("🔧 手動調整")
    if st.button("🔄 AI検出をやり直す"):
        keys_to_delete = ["keypoints", "model_loaded"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
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
        st.error(f"MediaPipeの初期化に失敗しました: {e}")
        return None

def safe_calculate_angle(p1, p2, p3):
    """安全な角度計算関数"""
    try:
        if not all(p is not None for p in [p1, p2, p3]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in [p1, p2, p3]):
            return None
        if not all(isinstance(coord, (int, float)) for p in [p1, p2, p3] for coord in p):
            return None
            
        a = np.array([float(p1[0]), float(p1[1])], dtype=np.float64)
        b = np.array([float(p2[0]), float(p2[1])], dtype=np.float64)
        c = np.array([float(p3[0]), float(p3[1])], dtype=np.float64)
        
        ab, cb = a - b, c - b
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        
        if ab_norm < 1e-10 or cb_norm < 1e-10:
            return None
            
        cosine = np.dot(ab, cb) / (ab_norm * cb_norm)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        
        if np.isnan(angle) or np.isinf(angle):
            return None
            
        return round(float(angle), 1)
    except Exception:
        return None

def safe_calculate_hip_ground_angle(hip_pos, knee_pos):
    """安全な股関節角度計算"""
    try:
        if not all(p is not None for p in [hip_pos, knee_pos]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in [hip_pos, knee_pos]):
            return None
            
        dx = float(knee_pos[0]) - float(hip_pos[0])
        dy = float(knee_pos[1]) - float(hip_pos[1])
        
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return None
            
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg = abs(angle_deg)
        elif angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return round(angle_deg, 1)
    except Exception:
        return None

def safe_vector_angle_with_ground(p1, p2):
    """安全なベクトル角度計算"""
    try:
        if not all(p is not None for p in [p1, p2]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in [p1, p2]):
            return None
            
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        
        v = np.array([dx, dy], dtype=np.float64)
        ground = np.array([1, 0], dtype=np.float64)
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            return None
            
        cos_theta = np.dot(v, ground) / norm_v
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        
        if np.isnan(angle) or np.isinf(angle):
            return None
            
        return round(float(angle), 1)
    except Exception:
        return None

def evaluate_angles(front_angle, rear_angle, front_hip_angle):
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

def evaluate_takeoff_angles(lower_angle, upper_angle, kunoji_angle):
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

def draw_skeleton_on_image(img, keypoints, joint_size, line_width):
    """画像に骨格を描画する関数"""
    try:
        # PILで描画
        img_pil = img.copy()
        draw = ImageDraw.Draw(img_pil)
        
        # 関節点の番号マッピング
        joint_numbers = {
            "LShoulder": "1", "RShoulder": "2", "LHip": "3", "RHip": "4",
            "LKnee": "5", "RKnee": "6", "LAnkle": "7", "RAnkle": "8", "C7": "9"
        }
        
        # 線を描画
        lines = [
            ("LShoulder", "LHip", "red"), ("LHip", "LKnee", "red"), ("LKnee", "LAnkle", "red"),
            ("RShoulder", "RHip", "red"), ("RHip", "RKnee", "red"), ("RKnee", "RAnkle", "red"),
            ("LShoulder", "RShoulder", "red"), ("LHip", "RHip", "red"),
        ]
        
        for a, b, color in lines:
            if a in keypoints and b in keypoints:
                x1, y1 = keypoints[a]
                x2, y2 = keypoints[b]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        
        # C7から骨盤への線
        if all(k in keypoints for k in ["C7", "RHip", "LHip", "RAnkle", "LAnkle"]):
            pelvis = keypoints["RHip"] if keypoints["RAnkle"][0] > keypoints["LAnkle"][0] else keypoints["LHip"]
            x1, y1 = keypoints["C7"]
            x2, y2 = pelvis
            draw.line([(x1, y1), (x2, y2)], fill="purple", width=line_width+2)
        
        # 関節点を描画
        for name, (x, y) in keypoints.items():
            # 外側の円（赤）
            draw.ellipse([x-joint_size, y-joint_size, x+joint_size, y+joint_size], 
                        fill="red", outline="darkred")
            # 内側の円（黄）
            inner_size = max(1, joint_size-2)
            draw.ellipse([x-inner_size, y-inner_size, x+inner_size, y+inner_size], 
                        fill="yellow", outline="orange")
            
            # 番号を描画
            if name in joint_numbers:
                number = joint_numbers[name]
                try:
                    # フォントサイズを調整
                    font_size = max(12, joint_size)
                    # デフォルトフォントを使用
                    draw.text((x+joint_size+2, y-joint_size), number, fill="white", anchor="lt")
                except Exception:
                    # フォント描画に失敗した場合はスキップ
                    pass
        
        return img_pil
    except Exception as e:
        st.error(f"画像描画エラー: {e}")
        return img

def manual_adjustment_interface(keypoints, img_width, img_height):
    """手動調整インターフェース"""
    joint_names_jp = {
        "LShoulder": "① 左肩", "RShoulder": "② 右肩",
        "LHip": "③ 左股関節", "RHip": "④ 右股関節", 
        "LKnee": "⑤ 左膝", "RKnee": "⑥ 右膝",
        "LAnkle": "⑦ 左足首", "RAnkle": "⑧ 右足首",
        "C7": "⑨ 第7頸椎"
    }
    
    st.subheader("🎯 関節点の手動調整")
    
    # 簡潔な調整インターフェース
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**上半身**")
        upper_joints = ["LShoulder", "RShoulder", "LHip", "RHip", "C7"]
        for joint in upper_joints:
            if joint in keypoints:
                current_x, current_y = keypoints[joint]
                st.write(f"**{joint_names_jp[joint]}**")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    new_x = st.number_input(
                        "X", min_value=0, max_value=img_width, 
                        value=int(current_x), step=1,
                        key=f"{joint}_x", label_visibility="visible"
                    )
                with col_y:
                    new_y = st.number_input(
                        "Y", min_value=0, max_value=img_height, 
                        value=int(current_y), step=1,
                        key=f"{joint}_y", label_visibility="visible"
                    )
                
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.rerun()
    
    with col2:
        st.write("**下半身**")
        lower_joints = ["LKnee", "RKnee", "LAnkle", "RAnkle"]
        for joint in lower_joints:
            if joint in keypoints:
                current_x, current_y = keypoints[joint]
                st.write(f"**{joint_names_jp[joint]}**")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    new_x = st.number_input(
                        "X", min_value=0, max_value=img_width, 
                        value=int(current_x), step=1,
                        key=f"{joint}_x2", label_visibility="visible"
                    )
                with col_y:
                    new_y = st.number_input(
                        "Y", min_value=0, max_value=img_height, 
                        value=int(current_y), step=1,
                        key=f"{joint}_y2", label_visibility="visible"
                    )
                
                if (new_x, new_y) != (current_x, current_y):
                    st.session_state.keypoints[joint] = (new_x, new_y)
                    st.rerun()

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
                        st.error(f"AI姿勢推定エラー: {e}")
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
                    st.stop()

        # レイアウト
        col_image, col_inputs = st.columns([2, 1])
        
        with col_image:
            st.subheader("🎯 骨格表示（関節点付き）")
            
            # 骨格描画
            skeleton_img = draw_skeleton_on_image(img, st.session_state.keypoints, joint_size, line_width)
            st.image(skeleton_img, use_container_width=True)

        with col_inputs:
            # 手動調整UI
            manual_adjustment_interface(st.session_state.keypoints, w, h)
            
            # 分析処理
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
                else:
                    st.warning("必要な関節点が不足しています")

    except Exception as e:
        st.error(f"🚨 エラーが発生しました: {str(e)}")
        st.write("エラーの詳細:")
        st.code(str(e))

else:
    st.info("📷 クラウチングスタートの画像をアップロードしてください。")
    st.markdown("""
    ### 🚀 完全エラー解決版の特徴
    - **キャンバスエラー完全解決**: streamlit-drawable-canvasを使用せず、PILで描画
    - **安定した動作**: NumPy配列比較エラーを根本的に解決
    - **直感的な調整**: 数値入力による精密な関節点調整
    - **リアルタイム更新**: 値変更で即座に画像更新
    - **高精度分析**: セット姿勢・飛び出し分析の両方に対応

    ### 📋 関節点番号
    - ① 左肩　② 右肩　③ 左股関節　④ 右股関節
    - ⑤ 左膝　⑥ 右膝　⑦ 左足首　⑧ 右足首
    - ⑨ 第7頸椎（C7）

    ### 🎯 使用方法
    1. 画像をアップロード
    2. AI自動検出の結果を確認
    3. 必要に応じて関節点を手動調整
    4. 分析結果とアドバイスを確認
    """)
