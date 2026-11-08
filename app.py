import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

# 安全匯入 OpenCV，避免 Python 3.14 雲端環境直接崩潰
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# 設定網頁版面配置
st.set_page_config(
    page_title="AI 輔助影像處理與人臉分析系統",
    page_icon="🖼️",
    layout="wide"
)

if not CV2_AVAILABLE:
    st.error("⚠️ 雲端環境提示：當前 Streamlit Cloud 的 Python 3.14 環境無法載入 OpenCV 套件。建議在您的「本機端 (Localhost)」執行以獲得完整的 100% 影像處理與人臉辨識功能！")

# 初始化 Session State 以確保 3x3 矩陣與範本雙向連動
if "kernel" not in st.session_state:
    st.session_state.kernel = np.array([[0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0]], dtype=float)

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

st.sidebar.header("📁 影像檔案上傳")
uploaded_file = st.sidebar.file_uploader("請上傳測試影像 (PNG, JPG, JPEG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None and CV2_AVAILABLE:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_name = uploaded_file.name
else:
    image = None
    img_name = "未上傳圖片"

st.markdown(
    '<h1 style="white-space: nowrap; font-size: 2.2rem;">🖼️ 基於 AI 輔助之互動式影像處理與人臉分析工具開發</h1>', 
    unsafe_allow_html=True
)
st.markdown("期中專題實作介面 ———— 整合色彩空間分析、手動卷積濾波與智慧人臉檢測")

if image is not None and CV2_AVAILABLE:
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 原始與中繼資訊", 
        "2. 色彩空間與直方圖", 
        "3. 卷積濾波與 PSNR", 
        "4. 智慧人臉偵測"
    ])

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

    with tab2:
        st.subheader("🎨 多色彩空間轉換與同步直方圖分析")
        color_mode = st.radio("選擇檢視模式：", ["RGB 三通道綜覽", "單通道顏色分離", "YUV 色彩空間解析"], horizontal=True)
        col_img, col_hist = st.columns(2)

        if color_mode == "RGB 三通道綜覽":
            with col_img:
                st.image(image_rgb, caption="RGB 原圖", use_column_width=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                for i, col in enumerate(('r', 'g', 'b')):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col, label=f"{col.upper()} 通道")
                ax.set_title("RGB 三通道像素強度分布直方圖")
                ax.legend()
                st.pyplot(fig)
        elif color_mode == "單通道顏色分離":
            channel_choice = st.selectbox("選擇要隔離的色彩通道：", ["紅色通道 (R)", "綠色通道 (G)", "藍色通道 (B)"])
            zeros = np.zeros_like(image[:, :, 0])
            if channel_choice == "紅色通道 (R)":
                isolated = cv2.merge([zeros, zeros, image[:, :, 2]])
                ch_idx, plot_color = 2, 'r'
            elif channel_choice == "綠色通道 (G)":
                isolated = cv2.merge([zeros, image[:, :, 1], zeros])
                ch_idx, plot_color = 1, 'g'
            else:
                isolated = cv2.merge([image[:, :, 0], zeros, zeros])
                ch_idx, plot_color = 0, 'b'
            with col_img:
                st.image(cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                hist = cv2.calcHist([image], [ch_idx], None, [256], [0, 256])
                ax.fill_between(range(256), hist.flatten(), color=plot_color, alpha=0.3)
                st.pyplot(fig)
        else:
            yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv_img)
            yuv_channel = st.selectbox("選擇 YUV 分量模式：", ["Y (亮度分量)", "U (色度分量)", "V (色度分量)"])
            target_chan = y if yuv_channel.startswith("Y") else (u if yuv_channel.startswith("U") else v)
            with col_img:
                st.image(target_chan, use_column_width=True, clamp=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                hist = cv2.calcHist([target_chan], [0], None, [256], [0, 256])
                ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.4)
                st.pyplot(fig)

    with tab3:
        st.subheader("⚙️ 3×3 卷積核矩陣自訂與 PSNR 品質評估")
        st.selectbox("快速濾波範本：", list(PRESETS.keys()), key="preset_selector", on_change=on_preset_change)
        k_cols = st.columns(3)
        for i in range(3):
            for j in range(3):
                with k_cols[j]:
                    st.session_state.kernel[i, j] = st.number_input(f"K[{i},{j}]", value=float(st.session_state.kernel[i, j]), step=0.1, key=f"k_{i}_{j}")
        auto_norm = st.checkbox("啟用自動正規化", value=True)
        kernel_to_apply = st.session_state.kernel.copy()
        if auto_norm and np.sum(kernel_to_apply) != 0:
            kernel_to_apply = kernel_to_apply / np.sum(kernel_to_apply)
        filtered_img = cv2.filter2D(image, -1, kernel_to_apply)
        psnr_val = cv2.PSNR(image, filtered_img)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(image_rgb, caption="原始影像", use_column_width=True)
        with col_f2:
            st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption=f"處理後 (PSNR: {psnr_val:.2f} dB)", use_column_width=True)
        st.metric("PSNR", f"{psnr_val:.2f} dB")

    with tab4:
        st.subheader("🤖 智慧人臉自動偵測與統計稽核")
        st.markdown("系統採用高門檻過濾機制，自動清除背景雜訊。")
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            try:
                urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", cascade_path)
            except Exception:
                pass
        faces = []
        try:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        except Exception:
            faces = []
        
        detected_img = image.copy()
        for idx, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(detected_img, f"Face #{idx}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        col_d1, col_d2 = st.columns([1.2, 1])
        with col_d1:
            st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), caption=f"人臉偵測結果 ({len(faces)} 張)", use_column_width=True)
        with col_d2:
            if len(faces) > 0:
                st.success(f"成功識別！總計：**{len(faces)} 張**")
                st.table([{"Face ID": f"#{i}", "X": x, "Y": y, "W": w, "H": h} for i, (x, y, w, h) in enumerate(faces, 1)])
            else:
                st.warning("未偵測到人臉。")
else:
    if CV2_AVAILABLE:
        st.warning("👈 請從左側側邊欄上傳圖片以啟動系統！")
