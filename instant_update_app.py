import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import math

# ページ設定
st.set_page_config(layout="wide", page_title="クラウチングスタート姿勢分析")

# セッション状態の初期化
def initialize_session_state():
    if "keypoints" not in st.session_state:
        st.session_state.keypoints = {}
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "last_canvas_data" not in st.session_state:
        st.session_state.last_canvas_data = None

initialize_session_state()

st.title("🏃 クラウチングスタート姿勢分析 & 飛び出し分析（完全修正版）")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    mode = st.selectbox("分析モード", ["セット姿勢", "飛び出し分析"])
    show_feedback = st.checkbox("フィードバック表示", value=True)
    joint_size = st.slider("関節点サイズ", 6, 20, 10)
    st.divider()
    st.header("🔧 手動調整")
    adjustment_mode = st.selectbox(
        "調整方法",
        ["プルダウン選択", "画像下部に横並び表示"]
    )
    if st.button("🔄 AI検出をやり直す"):
        for key in ["keypoints", "model_loaded", "last_canvas_data"]:
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
        # 入力値の検証
        if not all(p is not None for p in [p1, p2, p3]):
            return None
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in [p1, p2, p3]):
            return None
        if not all(isinstance(coord, (int, float)) for p in [p1, p2, p3] for coord in p):
            return None
            
        # numpy配列に変換（型を明示）
        a = np.array([float(p1[0]), float(p1[1])], dtype=np.float64)
        b = np.array([float(p2[0]), float(p2[1])], dtype=np.float64)
        c = np.array([float(p3[0]), float(p3[1])], dtype=np.float64)
        
        # ベクトル計算
        ab, cb = a - b, c - b
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        
        # ゼロ除算チェック
        if ab_norm < 1e-10 or cb_norm < 1e-10:
            return None
            
        # 角度計算
        cosine = np.dot(ab, cb) / (ab_norm * cb_norm)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        
        # 結果の検証
        if np.isnan(angle) or np.isinf(angle):
            return None
            
        return round(float(angle), 1)
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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

def keypoints_to_canvas_objects(keypoints, joint_size):
    """関節点をキャンバスオブジェクトに変換"""
    objects = []
    joint_numbers = {
        "LShoulder": "1", "RShoulder": "2", "LHip": "3", "RHip": "4",
        "LKnee": "5", "RKnee": "6", "LAnkle": "7", "RAnkle": "8", "C7": "9"
    }
    
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
    if "C7" in keypoints and "RHip" in keypoints and "LHip" in keypoints and "RAnkle" in keypoints and "LAnkle" in keypoints:
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
    
    return objects

def canvas_to_keypoints(objects, original_keypoints):
    """キャンバスオブジェクトを関節点に変換"""
    results = original_keypoints.copy()
    
    if objects is None:
        return results
        
    for obj in objects:
        if obj.get("type") == "circle" and "name" in obj:
            try:
                x = float(obj.get("left", 0))
                y = float(obj.get("top", 0))
                results[obj["name"]] = (int(x), int(y))
            except (ValueError, TypeError):
                continue
    
    return results

def safe_keypoints_comparison(dict1, dict2):
    """完全に安全な関節点比較関数"""
    try:
        # Noneチェック
        if dict1 is None or dict2 is None:
            return True
        
        # 型チェック
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            return True
        
        # キーの比較
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        if keys1 != keys2:
            return True
        
        # 各値の比較
        for key in keys1:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            # None値の処理
            if val1 is None and val2 is None:
                continue
            if val1 is None or val2 is None:
                return True
            
            # タプル/リストの比較
            try:
                if isinstance(val1, (tuple, list)) and isinstance(val2, (tuple, list)):
                    if len(val1) != len(val2):
                        return True
                    
                    # 数値比較（許容誤差付き）
                    for v1, v2 in zip(val1, val2):
                        diff = abs(float(v1) - float(v2))
                        if diff > 1.0:  # 1ピクセル以上の差があれば変更とみなす
                            return True
                else:
                    # その他の型
                    if str(val1) != str(val2):
                        return True
            except (ValueError, TypeError):
                # 変換エラーが発生した場合は変更ありとして扱う
                return True
        
        return False
    except Exception:
        # 何らかのエラーが発生した場合は変更ありとして扱う
        return True

def create_canvas_key():
    """キャンバス用のユニークキーを生成"""
    return f"canvas_{len(st.session_state.keypoints)}_{hash(str(st.session_state.keypoints)) % 10000}"

def manual_adjustment_dropdown(keypoints, img_width, img_height):
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
            st.rerun()

def manual_adjustment_horizontal(keypoints, img_width, img_height):
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
                                except:
                                    # デフォルト位置を設定
                                    st.session_state.keypoints[name] = (w//2, h//2)
                            
                            # C7（第7頸椎）の計算
                            if "LShoulder" in st.session_state.keypoints and "RShoulder" in st.session_state.keypoints:
                                lx, ly = st.session_state.keypoints["LShoulder"]
                                rx, ry = st.session_state.keypoints["RShoulder"]
                                st.session_state.keypoints["C7"] = ((lx+rx)//2, (ly+ry)//2)
                            
                            st.success("✅ AI検出完了！下記で関節点を調整してください。")
                        else:
                            st.warning("⚠️ AI検出に失敗しました。デフォルト位置を設定します。")
                            # デフォルト位置を設定
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
                        # デフォルト位置を設定
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
            st.subheader("🎯 画像上の関節点（ドラッグで移動可）")
            
            # キャンバス描画
            objects = keypoints_to_canvas_objects(st.session_state.keypoints, joint_size)
            canvas_key = create_canvas_key()
            
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
            
            # キャンバスからの更新を反映（完全修正版）
            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                try:
                    new_points = canvas_to_keypoints(canvas_result.json_data["objects"], st.session_state.keypoints)
                    
                    # 安全な比較を使用
                    if safe_keypoints_comparison(new_points, st.session_state.keypoints):
                        st.session_state.keypoints = new_points
                        # キャンバスデータを保存
                        st.session_state.last_canvas_data = canvas_result.json_data
                
                except Exception as e:
                    # キャンバス更新エラーは静かに無視
                    pass

        with col_inputs:
            # 手動調整UI
            if adjustment_mode == "プルダウン選択":
                manual_adjustment_dropdown(st.session_state.keypoints, w, h)
            else:
                manual_adjustment_horizontal(st.session_state.keypoints, w, h)
            
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
    ### 🚀 改良点・モード切替
    - **セット姿勢/飛び出し分析**: サイドバーで分析モードを切替
    - **即座更新**: 数値を変更するとリアルタイムで関節点が移動
    - **縦横表示**: X座標（横方向）、Y座標（縦方向）を明確に表示
    - **角度修正**: モードによって地面や「くの字」などの角度を計算
    - **番号表記**: 関節点の横に1,2,3...の番号を表示
    - **エラーハンドリング強化**: 安定性向上とエラー対策完備
    - **NumPy配列エラー完全修正**: 配列比較の問題を根本解決

    ### 📋 関節点番号
    - ① 左肩　② 右肩　③ 左股関節　④ 右股関節
    - ⑤ 左膝　⑥ 右膝　⑦ 左足首　⑧ 右足首
    - ⑨ 第7頸椎（C7）

    ### 📐 測定角度
    - **セット姿勢モード**
        - 前足の膝角度：股関節-膝-足首の角度
        - 後足の膝角度：股関節-膝-足首の角度  
        - 前足股関節角度：股関節から膝への線と地面の角度
    - **飛び出し分析モード**
        - 下半身角度：股関節→足首ベクトルと地面（進行方向）のなす角度
        - 上半身角度：第7頸椎(C7)→股関節ベクトルと地面（進行方向）のなす角度
        - くの字角度：第7頸椎(C7)-股関節-足首（股関節を頂点）
    """)
