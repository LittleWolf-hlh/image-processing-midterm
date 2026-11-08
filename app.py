import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 設定網頁版面配置
st.set_page_config(
    page_title="AI 輔助影像處理與人臉分析系統",
    page_icon="🖼️",
    layout="wide"
)

# 初始化 Session State 以確保 3x3 矩陣與範本雙向連動
if "kernel" not in st.session_state:
    st.session_state.kernel = np.array([[0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0]], dtype=float)

# 快速濾波範本定義
PRESETS = {
    "原始 (Identity)": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "高斯平滑 (Gaussian Blur)": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0,
    "銳化 (Sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "邊緣檢測 (Edge Detection)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "盒狀模糊 (Box Blur)": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float) / 9.0
}

def on_preset_change():
    selected = st.session_state.preset_selector
    if selected in PRESETS:
        st.session_state.kernel = PRESETS[selected].copy()

# 側邊欄：檔案上傳層
st.sidebar.header("📁 影像檔案上傳")
uploaded_file = st.sidebar.file_uploader("請上傳測試影像 (PNG, JPG, JPEG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

# 預設測試圖片處理或使用者上傳圖片
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_name = uploaded_file.name
else:
    image = None
    img_name = "未上傳圖片"

# 主標題
st.title("🖼️ 基於 AI 輔助之互動式影像處理與人臉分析工具開發")
st.markdown("期中專題實作介面 ———— 整合色彩空間分析、手動卷積濾波與智慧人臉檢測")

if image is not None:
    # 建立四大核心功能分頁 (st.tabs)
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 原始與中繼資訊", 
        "2. 色彩空間與直方圖", 
        "3. 卷積濾波與 PSNR", 
        "4. 智慧人臉偵測"
    ])

    # ----------------------------------------------------
    # 核心一：影像載入與中繼資訊
    # ----------------------------------------------------
    with tab1:
        st.subheader("📌 原始影像即時預覽與中繼資訊 (Metadata)")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption=f"Original Image: {img_name}", use_column_width=True)
        with col2:
            h, w, c = image.shape
            file_size_kb = len(file_bytes) / 1024 if uploaded_file else 0
            st.markdown("#### 📊 影像屬性摘要")
            st.info(f"""
            - **檔案名稱**：{img_name}
            - **解析度**：{w} × {h} 像素
            - **色彩維度**：{c} 頻道 (RGB)
            - **檔案大小**：{file_size_kb:.2f} KB
            """)
            st.markdown("💡 **提示**：您可以透過左側側邊欄隨時更換不同照片以進行即時互動分析。")

    # ----------------------------------------------------
    # 核心二：多色彩空間與同步直方圖分析
    # ----------------------------------------------------
    with tab2:
        st.subheader("🎨 多色彩空間轉換與同步直方圖分析")
        color_mode = st.radio("選擇檢視模式：", ["RGB 三通道綜覽", "單通道顏色分離", "YUV 色彩空間解析"], horizontal=True)

        col_img, col_hist = st.columns(2)

        if color_mode == "RGB 三通道綜覽":
            with col_img:
                st.image(image_rgb, caption="RGB 原圖", use_column_width=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                colors = ('r', 'g', 'b')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col, label=f"{col.upper()} 通道")
                ax.set_title("RGB 三通道像素強度分布直方圖")
                ax.set_xlabel("Pixel Intensity (0-255)")
                ax.set_ylabel("Count")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

        elif color_mode == "單通道顏色分離":
            channel_choice = st.selectbox("選擇要隔離的色彩通道：", ["紅色通道 (R)", "綠色通道 (G)", "藍色通道 (B)"])
            zeros = np.zeros_like(image[:, :, 0])
            if channel_choice == "紅色通道 (R)":
                isolated = cv2.merge([zeros, zeros, image[:, :, 2]])
                ch_idx, ch_name, plot_color = 2, "Red", 'r'
            elif channel_choice == "綠色通道 (G)":
                isolated = cv2.merge([zeros, image[:, :, 1], zeros])
                ch_idx, ch_name, plot_color = 1, "Green", 'g'
            else:
                isolated = cv2.merge([image[:, :, 0], zeros, zeros])
                ch_idx, ch_name, plot_color = 0, "Blue", 'b'

            with col_img:
                st.image(cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB), caption=f"{channel_choice} 隔離影像", use_column_width=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                hist = cv2.calcHist([image], [ch_idx], None, [256], [0, 256])
                ax.fill_between(range(256), hist.flatten(), color=plot_color, alpha=0.3)
                ax.plot(hist, color=plot_color)
                ax.set_title(f"{ch_name} Channel Histogram")
                ax.set_xlabel("Pixel Intensity (0-255)")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

        else:  # YUV 色彩空間解析
            yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv_img)
            yuv_channel = st.selectbox("選擇 YUV 分量模式：", ["Y (亮度分量)", "U (色度分量)", "V (色度分量)"])
            
            if yuv_channel.startswith("Y"):
                target_chan, title_str = y, "Y Channel (Brightness)"
            elif yuv_channel.startswith("U"):
                target_chan, title_str = u, "U Channel (Chrominance)"
            else:
                target_chan, title_str = v, "V Channel (Chrominance)"

            with col_img:
                st.image(target_chan, caption=yuv_channel, use_column_width=True, clamp=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                hist = cv2.calcHist([target_chan], [0], None, [256], [0, 256])
                ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.4)
                ax.plot(hist, color='black')
                ax.set_title(title_str)
                ax.set_xlabel("Pixel Intensity (0-255)")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

    # ----------------------------------------------------
    # 核心三：自訂卷積核矩陣與 PSNR 評估
    # ----------------------------------------------------
    with tab3:
        st.subheader("⚙️ 3×3 卷積核矩陣自訂與 PSNR 品質評估")
        
        st.selectbox(
            "快速濾波範本 (選擇後將即時連動下方 9 個輸入框)：",
            list(PRESETS.keys()),
            key="preset_selector",
            on_change=on_preset_change
        )

        st.markdown("#### 🔢 3×3 卷積核矩陣手動編輯器")
        k_cols = st.columns(3)
        for i in range(3):
            for j in range(3):
                with k_cols[j]:
                    st.session_state.kernel[i, j] = st.number_input(
                        f"K[{i},{j}]",
                        value=float(st.session_state.kernel[i, j]),
                        step=0.1,
                        format="%.2f",
                        key=f"k_{i}_{j}"
                    )

        auto_norm = st.checkbox("啟用自動正規化 (Normalization，避免影像過曝)", value=True)

        kernel_to_apply = st.session_state.kernel.copy()
        if auto_norm:
            k_sum = np.sum(kernel_to_apply)
            if k_sum != 0:
                kernel_to_apply = kernel_to_apply / k_sum

        filtered_img = cv2.filter2D(image, -1, kernel_to_apply)
        psnr_val = cv2.PSNR(image, filtered_img)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(image_rgb, caption="原始影像", use_column_width=True)
        with col_f2:
            st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption=f"濾波處理後影像 (PSNR: {psnr_val:.2f} dB)", use_column_width=True)

        st.metric(label="📈 峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)", value=f"{psnr_val:.2f} dB")

    # ----------------------------------------------------
    # 核心四：智慧人臉自動偵測與統計
    # ----------------------------------------------------
    with tab4:
        st.subheader("🤖 智慧人臉自動偵測與統計稽核")
        st.markdown("系統採用高門檻過濾機制，自動清除背景與衣服紋理等誤判雜訊。")

        # 檢查 OpenCV 是否包含 CascadeClassifier 屬性（防範雲端環境編譯缺失）
        if not hasattr(cv2, 'CascadeClassifier'):
            st.warning("⚠️ 雲端環境提醒：目前的雲端 Python 執行環境未包含 `CascadeClassifier` 模組。本機端執行時功能完全正常。")
        else:
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

            face_cascade = cv2.CascadeClassifier(cascade_path)

            if face_cascade.empty():
                st.error("⚠️ 錯誤：無法載入 Haar-Cascade 人臉分類器 XML 模型檔！請確認專案根目錄是否包含 `haarcascade_frontalface_default.xml`。")
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=8,
                    minSize=(80, 80)
                )

                detected_img = image.copy()
                for idx, (x, y, w, h) in enumerate(faces, start=1):
                    cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(detected_img, f"Face #{idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                col_d1, col_d2 = st.columns([1.2, 1])
                with col_d1:
                    st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), caption=f"人臉偵測結果 (發現 {len(faces)} 張人臉)", use_column_width=True)
                with col_d2:
                    st.markdown("#### 📋 偵測結果摘要")
                    st.success(f"成功精準識別並標記人臉！總計偵測數：**{len(faces)} 張**")

                    if len(faces) > 0:
                        face_data = []
                        for idx, (x, y, w, h) in enumerate(faces, start=1):
                            face_data.append({
                                "Face ID": f"#{idx}",
                                "X 座標": x,
                                "Y 座標": y,
                                "寬度 (W)": w,
                                "高度 (H)": h
                            })
                        st.table(face_data)
                    else:
                        st.warning("畫面中未偵測到符合門檻的人臉，請嘗試更換測試照片。")

else:
    st.warning("👈 請從左側側邊欄上傳圖片，以啟動系統四大核心互動模組！")
