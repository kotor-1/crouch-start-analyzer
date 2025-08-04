# アプリケーションの改善案：

# 1. 分析結果保存機能
def save_analysis_results(data, filename="analysis_results.csv"):
    """分析結果をCSVファイルに保存する"""
    import pandas as pd
    import datetime
    
    # タイムスタンプ付きでデータをフォーマット
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["timestamp"] = timestamp
    
    # DataFrameの作成
    df = pd.DataFrame([data])
    
    # ファイルが存在するかチェックして追加または新規作成
    try:
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(filename, index=False)
    except FileNotFoundError:
        df.to_csv(filename, index=False)
    
    return filename

# 2. 画像比較機能
def setup_comparison_mode():
    """ビフォー/アフター画像比較のためのUI要素を追加"""
    st.subheader("📊 比較モード")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**ビフォー画像**")
        before_file = st.file_uploader("前回の画像", type=["png", "jpg", "jpeg"], key="before")
        
    with col2:
        st.write("**アフター画像**")
        after_file = st.file_uploader("今回の画像", type=["png", "jpg", "jpeg"], key="after")
    
    return before_file, after_file

# 3. 時間経過による進捗追跡
def show_progress_chart(data_file="analysis_results.csv"):
    """保存された分析データから進捗グラフを表示"""
    import pandas as pd
    import plotly.express as px
    
    try:
        df = pd.read_csv(data_file)
        if len(df) > 1:
            st.subheader("💹 経過分析")
            
            # 時間経過によるメトリクスのプロット
            fig = px.line(
                df, x="timestamp", 
                y=["front_angle", "rear_angle", "front_hip_angle"],
                title="姿勢指標の変化",
                labels={"value": "角度 (度)", "timestamp": "日時"},
                markers=True
            )
            
            st.plotly_chart(fig)
            
            return True
        return False
    except Exception:
        return False

# 4. 理想的な姿勢のビジュアルガイド
def draw_ideal_positions(canvas_data, keypoints):
    """理想的な関節位置と角度を示すビジュアルガイドを追加"""
    objects = canvas_data.copy()
    
    # 主要関節の理想的な角度指標を追加
    ideal_angles = {
        "front_knee": 90,
        "rear_knee": 127.5,  # 120-135の中間点
        "front_hip": 50      # 40-60の中間点
    }
    
    # （実装ではキャンバスに視覚的な弧やガイドを追加します）
    
    return objects

# 5. 具体的なトレーニング推奨を含む強化されたフィードバック
def get_detailed_recommendations(angles):
    """測定された角度に基づいて具体的なトレーニング推奨を提供"""
    recommendations = []
    
    if angles["front_angle"] < 85:
        recommendations.append({
            "issue": "前足の膝角度が狭すぎます",
            "training": "壁スクワット: 壁に背を向けて立ち、90度の角度でスクワット姿勢を保持する練習",
            "frequency": "1日3セット、各30秒間"
        })
    elif angles["front_angle"] > 95:
        recommendations.append({
            "issue": "前足の膝角度が広すぎます",
            "training": "ディープスクワット: 通常より深くスクワットを行い、適切な膝の曲げ方を感覚的に覚える",
            "frequency": "1日3セット、各10回"
        })
    
    # その他の関節に関する推奨事項...
    
    return recommendations

