import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
from scipy.ndimage import label, find_objects

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
    file_bytes = uploaded_file.getvalue()
    image_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image_pil)
    img_name = uploaded_file.name
else:
    image_pil = None
    image_np = None
    img_name = "未上傳圖片"

# 主標題 (強制不換行對齊處理)
st.markdown(
    '<h1 style="white-space: nowrap; font-size: 2.2rem;">🖼️ 基於 AI 輔助之互動式影像處理與人臉分析工具開發</h1>', 
    unsafe_allow_html=True
)
st.markdown("期中專題實作介面 ———— 整合色彩空間分析、手動卷積濾波與動態視覺特徵偵測")

if image_np is not None:
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
            st.image(image_pil, caption=f"Original Image: {img_name}", use_column_width=True)
        with col2:
            h, w, c = image_np.shape
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
                st.image(image_pil, caption="RGB 原圖", use_column_width=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                colors = ('r', 'g', 'b')
                for i, col in enumerate(colors):
                    hist, _ = np.histogram(image_np[:, :, i], bins=256, range=(0, 256))
                    ax.plot(hist, color=col, label=f"{col.upper()} 通道")
                ax.set_title("RGB 三通道像素強度分布直方圖")
                ax.set_xlabel("Pixel Intensity (0-255)")
                ax.set_ylabel("Count")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

        elif color_mode == "單通道顏色分離":
            channel_choice = st.selectbox("選擇要隔離的色彩通道：", ["紅色通道 (R)", "綠色通道 (G)", "藍色通道 (B)"])
            isolated_np = np.zeros_like(image_np)
            if channel_choice == "紅色通道 (R)":
                isolated_np[:, :, 0] = image_np[:, :, 0]
                ch_idx, ch_name, plot_color = 0, "Red", 'r'
            elif channel_choice == "綠色通道 (G)":
                isolated_np[:, :, 1] = image_np[:, :, 1]
                ch_idx, ch_name, plot_color = 1, "Green", 'g'
            else:
                isolated_np[:, :, 2] = image_np[:, :, 2]
                ch_idx, ch_name, plot_color = 2, "Blue", 'b'

            isolated_pil = Image.fromarray(isolated_np)
            with col_img:
                st.image(isolated_pil, caption=f"{channel_choice} 隔離影像", use_column_width=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                hist, _ = np.histogram(image_np[:, :, ch_idx], bins=256, range=(0, 256))
                ax.fill_between(range(256), hist, color=plot_color, alpha=0.3)
                ax.plot(hist, color=plot_color)
                ax.set_title(f"{ch_name} Channel Histogram")
                ax.set_xlabel("Pixel Intensity (0-255)")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

        else:  # YUV 色彩空間解析
            r = image_np[:, :, 0].astype(float)
            g = image_np[:, :, 1].astype(float)
            b = image_np[:, :, 2].astype(float)
            y_chan = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
            u_chan = np.clip(-0.14713 * r - 0.28886 * g + 0.436 * b + 128, 0, 255).astype(np.uint8)
            v_chan = np.clip(0.615 * r - 0.51499 * g - 0.10001 * b + 128, 0, 255).astype(np.uint8)

            yuv_channel = st.selectbox("選擇 YUV 分量模式：", ["Y (亮度分量)", "U (色度分量)", "V (色度分量)"])
            if yuv_channel.startswith("Y"):
                target_chan, title_str = y_chan, "Y Channel (Brightness)"
            elif yuv_channel.startswith("U"):
                target_chan, title_str = u_chan, "U Channel (Chrominance)"
            else:
                target_chan, title_str = v_chan, "V Channel (Chrominance)"

            with col_img:
                st.image(target_chan, caption=yuv_channel, use_column_width=True, clamp=True)
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                hist, _ = np.histogram(target_chan, bins=256, range=(0, 256))
                ax.fill_between(range(256), hist, color='gray', alpha=0.4)
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

        # 執行 3x3 卷積濾波
        img_float = image_np.astype(float)
        h, w, c = img_float.shape
        padded = np.pad(img_float, ((1, 1), (1, 1), (0, 0)), mode='edge')
        filtered_float = np.zeros_like(img_float)
        for pi in range(3):
            for pj in range(3):
                filtered_float += padded[pi:pi+h, pj:pj+w, :] * kernel_to_apply[pi, pj]
        filtered_np = np.clip(filtered_float, 0, 255).astype(np.uint8)
        filtered_pil = Image.fromarray(filtered_np)

        # 計算 PSNR
        mse = np.mean((image_np.astype(float) - filtered_float) ** 2)
        psnr_val = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(image_pil, caption="原始影像", use_column_width=True)
        with col_f2:
            st.image(filtered_pil, caption=f"濾波處理後影像 (PSNR: {psnr_val:.2f} dB)", use_column_width=True)

        st.metric(label="📈 峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)", value=f"{psnr_val:.2f} dB")

    # ----------------------------------------------------
    # 核心四：動態特徵與膚色區域偵測 (即時演算法)
    # ----------------------------------------------------
    with tab4:
        st.subheader("🤖 動態視覺特徵與區域偵測稽核")
        st.markdown("系統採用 **YCbCr 色彩空間膚色分割與連通區域分析 (Blob Analysis)**，對上傳的圖片進行即時動態特徵擷取。")

        # 1. 轉換為 YCbCr 色彩空間進行即時膚色與特徵檢測
        r = image_np[:, :, 0].astype(float)
        g = image_np[:, :, 1].astype(float)
        b = image_np[:, :, 2].astype(float)
        
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        y_luma = 0.299 * r + 0.587 * g + 0.114 * b

        # 2. 動態遮罩：過濾出符合膚色與明度特徵的候選區域
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (y_luma > 50)

        # 3. 連通區域標記與物件萃取 (Blob Detection)
        labeled_array, num_features = label(skin_mask)
        objects = find_objects(labeled_array)

        faces = []
        h_img, w_img = image_np.shape[:2]
        min_size = max(20, int(w_img * 0.03))
        max_size = int(w_img * 0.4)

        for obj in objects:
            if obj is not None:
                slice_y, slice_x = obj
                y_min, y_max = slice_y.start, slice_y.stop
                x_min, x_max = slice_x.start, slice_x.stop
                box_h = y_max - y_min
                box_w = x_max - x_min
                # 過濾雜訊，保留符合臉部比例的動態區塊
                if min_size < box_h < max_size and min_size < box_w < max_size:
                    aspect_ratio = box_w / box_h
                    if 0.5 < aspect_ratio < 2.0:
                        faces.append((x_min, y_min, box_w, box_h))

        # 繪製動態偵測結果
        draw_img = image_pil.copy()
        draw = ImageDraw.Draw(draw_img)

        for idx, (x, y, w_box, h_box) in enumerate(faces, start=1):
            draw.rectangle([x, y, x + w_box, y + h_box], outline=(0, 255, 0), width=3)
            draw.text((x, max(0, y - 15)), f"Region #{idx}", fill=(0, 255, 0))

        col_d1, col_d2 = st.columns([1.2, 1])
        with col_d1:
            st.image(draw_img, caption=f"動態特徵偵測結果 (即時識別 {len(faces)} 個特徵區域)", use_column_width=True)
        with col_d2:
            st.markdown("#### 📋 即時演算法偵測摘要")
            if len(faces) > 0:
                st.success(f"成功透過 YCbCr 演算法即時分割並標記特徵！總計偵測數：**{len(faces)} 個**")
                face_data = []
                for idx, (x, y, w_box, h_box) in enumerate(faces, start=1):
                    face_data.append({
                        "Region ID": f"#{idx}",
                        "X 座標": x,
                        "Y 座標": y,
                        "寬度 (W)": w_box,
                        "高度 (H)": h_box
                    })
                st.table(face_data)
            else:
                st.warning("未在畫面中檢測到符合門檻的特徵區塊，請嘗試更換對比清晰的測試照片。")

else:
    st.warning("👈 請從左側側邊欄上傳圖片，以啟動系統四大核心互動模組！")
