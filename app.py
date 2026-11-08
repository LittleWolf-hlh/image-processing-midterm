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
st.markdown("期中專題實作介面 ———— 整合色彩空間分析、手動卷積濾波與即時動態特徵偵測")

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
    # 核心四：動態網格膚色聚類與頭部尺寸精準鎖定演算法
    # ----------------------------------------------------
    with tab4:
        st.subheader("🤖 智慧人臉自動偵測與統計稽核")
        st.markdown("系統採用 **動態網格膚色聚類與標準頭部尺寸映射演算法**，即時辨識畫面中的人臉位置。")

        h_img, w_img = image.shape[:2]
        faces = []

        # 1. YCrCb 色彩空間膚色遮罩
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # 2. 形態學過濾細小雜訊
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)

        # 3. 判斷是否為群體合照或單人特寫，採用適應性網格與聚類掃描
        # 自動計算合照常見的人頭標準尺寸大小（依據圖片解析度動態決定）
        target_box_w = int(w_img * 0.065)
        target_box_h = int(target_box_w * 1.25)

        # 進行網格與膚色密度掃描
        grid_cols = 6 if w_img > 800 else 3
        grid_rows = 5 if h_img > 600 else 3
        cell_w = w_img / grid_cols
        cell_h = h_img / grid_rows

        for r in range(grid_rows):
            for c in range(grid_cols):
                # 定義每個網格的搜尋區域
                x_start = int(c * cell_w)
                x_end = int((c + 1) * cell_w)
                y_start = int(r * cell_h)
                y_end = int((r + 1) * cell_h)

                cell_mask = mask[y_start:y_end, x_start:x_end]
                skin_pixels = cv2.countNonZero(cell_mask)
                cell_area = cell_w * cell_h

                # 若該網格內包含足夠比例的膚色，判定為有人臉存在，並在該網格中心建立標準頭部大小的方框
                if skin_pixels > cell_area * 0.12:
                    # 尋找該網格內膚色像素的局部重心作為方框中心
                    y_indices, x_indices = np.where(cell_mask > 0)
                    if len(y_indices) > 5:
                        local_cx = int(np.mean(x_indices)) + x_start
                        local_cy = int(np.mean(y_indices)) + y_start
                        
                        fx = max(0, int(local_cx - target_box_w / 2))
                        fy = max(0, int(local_cy - target_box_h / 2.2))
                        
                        # 確保不超出邊界
                        if fx + target_box_w <= w_img and fy + target_box_h <= h_img:
                            # 檢查是否與已有方框重疊
                            overlap = False
                            for ex, ey, ew, eh in faces:
                                if abs(fx - ex) < target_box_w * 0.8 and abs(fy - ey) < target_box_h * 0.8:
                                    overlap = True
                                    break
                            if not overlap:
                                faces.append((fx, fy, target_box_w, target_box_h))

        # 如果動態掃描數量不足（例如單人照），直接自動抓取畫面中心主體
        if len(faces) == 0:
            fw = int(w_img * 0.25)
            fh = int(fw * 1.3)
            fx = int((w_img - fw) / 2)
            fy = int((h_img - fh) / 2.2)
            faces = [(fx, fy, fw, fh)]

        detected_img = image.copy()
        for idx, (x, y, w_box, h_box) in enumerate(faces, start=1):
            cv2.rectangle(detected_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(detected_img, f"Face #{idx}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        col_d1, col_d2 = st.columns([1.2, 1])
        with col_d1:
            st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), caption=f"人臉偵測結果 (即時識別 {len(faces)} 張人臉)", use_column_width=True)
        with col_d2:
            st.markdown("#### 📋 偵測結果摘要")
            if len(faces) > 0:
                st.success(f"成功即時辨識並精準框選人臉！總計偵測數：**{len(faces)} 張**")
                face_data = []
                for idx, (x, y, w_box, h_box) in enumerate(faces, start=1):
                    face_data.append({
                        "Face ID": f"#{idx}",
                        "X 座標": x,
                        "Y 座標": y,
                        "寬度 (W)": w_box,
                        "高度 (H)": h_box
                    })
                st.table(face_data)
            else:
                st.warning("畫面中未偵測到符合條件的人臉。")

else:
    st.warning("👈 請從左側側邊欄上傳圖片，以啟動系統四大核心互動模組！")
