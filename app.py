import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

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

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_name = uploaded_file.name
else:
    image = None
    img_name = "未上傳圖片"

# 主標題 (強制不換行對齊處理)
st.markdown(
    '<h1 style="white-space: nowrap; font-size: 2.2rem;">🖼️ 基於 AI 輔助之互動式影像處理與人臉分析工具開發</h1>', 
    unsafe_allow_html=True
)
st.markdown("期中專題實作介面 ———— 整合色彩空間分析、手動卷積濾波與智慧動態特徵偵測")

if image is not None:
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
            st.markdown("💡 **提示**：系統已成功載入並運行於雲端即時運算環境。")

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
    # 核心四：頭部精準切片與非極大值抑制演算法
    # ----------------------------------------------------
    with tab4:
        st.subheader("🤖 智慧人臉自動偵測與統計稽核")
        st.markdown("系統採用 **YCrCb 膚色空間與頭部特徵精準切片演算法**，自動鎖定每個人臉位置而不延伸至身體。")

        h_img, w_img = image.shape[:2]
        faces = []

        # 1. YCrCb 色彩空間膚色遮罩
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # 2. 形態學雜訊過濾
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_candidates = []
        min_area = (w_img * h_img) * 0.0008
        max_area = (w_img * h_img) * 0.35

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h / float(w)
                if 0.5 < aspect_ratio < 3.5:
                    # 關鍵優化：將檢測到的區塊嚴格轉換為獨立的「頭部/臉部」比例方框（鎖定在頂端）
                    face_w = w
                    face_h = int(w * 1.1)  # 保持標準臉部比例
                    face_x = x
                    face_y = y             # 從頂端開始，絕不延伸到胸口
                    raw_candidates.append((face_x, face_y, face_w, face_h))

        # 3. 非極大值抑制（NMS）與距離過濾：確保每張臉只有一個精準框
        raw_candidates = sorted(raw_candidates, key=lambda b: b[2] * b[3], reverse=True)
        for box in raw_candidates:
            x, y, w, h = box
            overlap = False
            for fx, fy, fw, fh in faces:
                if abs(x - fx) < fw * 0.5 and abs(y - fy) < fh * 0.5:
                    overlap = True
                    break
            if not overlap and 0 <= x < w_img and 0 <= y < h_img:
                # 確保邊界不超出圖片
                final_w = min(w, w_img - x)
                final_h = min(h, h_img - y)
                faces.append((x, y, final_w, final_h))

        detected_img = image.copy()
        for idx, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(detected_img, f"Face #{idx}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        col_d1, col_d2 = st.columns([1.2, 1])
        with col_d1:
            st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), caption=f"人臉偵測結果 (即時識別 {len(faces)} 張人臉)", use_column_width=True)
        with col_d2:
            st.markdown("#### 📋 偵測結果摘要")
            if len(faces) > 0:
                st.success(f"成功即時辨識並精準框選人臉！總計偵測數：**{len(faces)} 張**")
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
                st.warning("畫面中未偵測到符合條件的人臉。")

else:
    st.warning("👈 請從左側側邊欄上傳圖片，以啟動系統四大核心互動模組！")
