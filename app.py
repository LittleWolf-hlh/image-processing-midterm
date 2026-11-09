from io import BytesIO
import math
import os
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 頁面基本配置
st.set_page_config(
    page_title="AI 輔助影像處理與人臉分析工具",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource
def get_yunet_onnx_model_path() -> str:
    """下載並快取 OpenCV 官方 YuNet ONNX 人臉偵測模型，並驗證檔案正確性"""
    onnx_path = "face_detection_yunet_2023mar.onnx"
    onnx_url = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    need_download = False
    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 50000:
        need_download = True
    else:
        # 檢查檔案是否誤下載成 HTML 錯誤網頁
        with open(onnx_path, 'rb') as f:
            header_snippet = f.read(100)
            if b"<!DOCTYPE html>" in header_snippet or b"<html" in header_snippet:
                need_download = True

    if need_download:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(onnx_url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp, open(onnx_path, 'wb') as f:
                f.write(resp.read())
        except Exception as e:
            st.error(f"下載 YuNet 模型失敗: {e}")

    return os.path.abspath(onnx_path)


st.title("📷 基於 AI 輔助之互動式影像處理與人臉分析工具")
st.write("期中專題實作介面 —— 整合色彩空間分析、自訂卷積濾波與 AI 深度學習人臉檢測")

# -------------------------------------------------------------
# 核心一：影像載入與基礎解析
# -------------------------------------------------------------
st.sidebar.header("影像載入與解析")
uploaded_file = st.sidebar.file_uploader(
    "請上傳測試影像", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, channels = img_bgr.shape
            file_size_kb = uploaded_file.size / 1024

            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 影像中繼資訊 (Metadata)")
            st.sidebar.write(f"- **檔案名稱**: {uploaded_file.name}")
            st.sidebar.write(f"- **解析度**: {width} × {height} 像素")
            st.sidebar.write(f"- **色彩維度**: {channels} 頻道 (RGB)")
            st.sidebar.write(f"- **檔案大小**: {file_size_kb:.2f} KB")

            # 建立四大核心分頁
            tab1, tab2, tab3, tab4 = st.tabs([
                "基礎資訊",
                "1. 色彩空間與動態直方圖",
                "2. 自訂卷積與 PSNR",
                "3. AI 人臉偵測",
            ])

            # --- Tab 1: 基礎資訊 ---
            with tab1:
                st.subheader("圖片預覽與中繼資訊")
                st.image(
                    img_rgb,
                    caption=f"原始影像預覽: {uploaded_file.name}",
                    use_container_width=True,
                )
                res_col, chan_col, size_col = st.columns(3)
                res_col.metric("解析度（Width × Height）", f"{width} × {height}")
                chan_col.metric("色彩通道數（Channels）", str(channels))
                size_col.metric("檔案大小", f"{file_size_kb:.2f} KB")

            # --- Tab 2: 多色彩空間轉換與動態直方圖分析 ---
            with tab2:
                st.subheader("多色彩空間切換與動態直方圖分析")
                mode = st.radio(
                    "選擇檢視模式",
                    ["Original (RGB)", "RGB 獨立通道", "YUV 色彩空間"],
                    horizontal=True
                )

                if mode == "Original (RGB)":
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    ax[0].imshow(img_rgb)
                    ax[0].set_title("Original RGB Image")
                    ax[0].axis("off")

                    colors = ('r', 'g', 'b')
                    for i, col in enumerate(colors):
                        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                        ax[1].plot(hist, color=col, label=f'Channel {col.upper()}')
                    ax[1].set_xlim([0, 256])
                    ax[1].set_title("RGB Histogram")
                    ax[1].set_xlabel("Pixel Intensity (0-255)")
                    ax[1].set_ylabel("Pixel Count")
                    ax[1].legend()
                    st.pyplot(fig)
                    plt.close(fig)

                elif mode == "RGB 獨立通道":
                    img_r = np.zeros_like(img_rgb)
                    img_r[:, :, 0] = img_rgb[:, :, 0]

                    img_g = np.zeros_like(img_rgb)
                    img_g[:, :, 1] = img_rgb[:, :, 1]

                    img_b = np.zeros_like(img_rgb)
                    img_b[:, :, 2] = img_rgb[:, :, 2]

                    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

                    channels_data = [
                        ("Red Channel", img_r, img_rgb[:, :, 0], 'red'),
                        ("Green Channel", img_g, img_rgb[:, :, 1], 'green'),
                        ("Blue Channel", img_b, img_rgb[:, :, 2], 'blue')
                    ]
                    for idx, (title, color_img, single_ch, color_code) in enumerate(channels_data):
                        axes[0, idx].imshow(color_img)
                        axes[0, idx].set_title(title)
                        axes[0, idx].axis("off")

                        hist = cv2.calcHist([single_ch], [0], None, [256], [0, 256])
                        axes[1, idx].plot(hist, color=color_code)
                        axes[1, idx].set_xlim([0, 256])
                        axes[1, idx].set_title(f"{title} Histogram")
                        axes[1, idx].set_xlabel("Pixel Intensity")
                        axes[1, idx].set_ylabel("Pixel Count")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                elif mode == "YUV 色彩空間":
                    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                    Y, U, V = img_yuv[:, :, 0], img_yuv[:, :, 1], img_yuv[:, :, 2]

                    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
                    yuv_data = [
                        ("Y Component (Luminance)", Y),
                        ("U Component (Chrominance)", U),
                        ("V Component (Chrominance)", V)
                    ]

                    for idx, (title, ch_img) in enumerate(yuv_data):
                        axes[0, idx].imshow(ch_img, cmap='gray')
                        axes[0, idx].set_title(title)
                        axes[0, idx].axis("off")

                        hist = cv2.calcHist([ch_img], [0], None, [256], [0, 256])
                        axes[1, idx].plot(hist, color='black')
                        axes[1, idx].set_xlim([0, 256])
                        axes[1, idx].set_title(f"{title} Histogram")
                        axes[1, idx].set_xlabel("Pixel Intensity")
                        axes[1, idx].set_ylabel("Pixel Count")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            # --- Tab 3: 自訂卷積與 PSNR 評估 ---
            with tab3:
                st.subheader("自訂卷積核 (Kernel Matrix) 空間濾波與 PSNR 分析")

                if "m_00" not in st.session_state:
                    st.session_state.m_00 = 0.0
                    st.session_state.m_01 = -1.0
                    st.session_state.m_02 = 0.0
                    st.session_state.m_10 = -1.0
                    st.session_state.m_11 = 5.0
                    st.session_state.m_12 = -1.0
                    st.session_state.m_20 = 0.0
                    st.session_state.m_21 = -1.0
                    st.session_state.m_22 = 0.0
                    st.session_state.norm_chk = False

                def on_preset_change():
                    preset = st.session_state.k_preset
                    if preset == "高斯平滑 (Gaussian Blur)":
                        vals = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
                        st.session_state.norm_chk = True
                    elif preset == "拉普拉斯銳化 (Sharpen)":
                        vals = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]
                        st.session_state.norm_chk = False
                    elif preset == "邊緣檢測 (Edge Detection)":
                        vals = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
                        st.session_state.norm_chk = False
                    elif preset == "浮雕效果 (Emboss)":
                        vals = [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]
                        st.session_state.norm_chk = False
                    else:
                        vals = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
                        st.session_state.norm_chk = False

                    st.session_state.m_00 = vals[0][0]
                    st.session_state.m_01 = vals[0][1]
                    st.session_state.m_02 = vals[0][2]
                    st.session_state.m_10 = vals[1][0]
                    st.session_state.m_11 = vals[1][1]
                    st.session_state.m_12 = vals[1][2]
                    st.session_state.m_20 = vals[2][0]
                    st.session_state.m_21 = vals[2][1]
                    st.session_state.m_22 = vals[2][2]

                st.selectbox(
                    "快速選擇預設濾波核",
                    [
                        "手動自訂矩陣",
                        "拉普拉斯銳化 (Sharpen)",
                        "高斯平滑 (Gaussian Blur)",
                        "邊緣檢測 (Edge Detection)",
                        "浮雕效果 (Emboss)",
                    ],
                    key="k_preset",
                    on_change=on_preset_change,
                )

                st.markdown("#### 🛠️ 3×3 卷積核矩陣手動編輯器")
                k_col1, k_col2, k_col3 = st.columns(3)
                with k_col1:
                    r0c0 = st.number_input("K[0,0]", step=1.0, key="m_00")
                    r1c0 = st.number_input("K[1,0]", step=1.0, key="m_10")
                    r2c0 = st.number_input("K[2,0]", step=1.0, key="m_20")
                with k_col2:
                    r0c1 = st.number_input("K[0,1]", step=1.0, key="m_01")
                    r1c1 = st.number_input("K[1,1]", step=1.0, key="m_11")
                    r2c1 = st.number_input("K[2,1]", step=1.0, key="m_21")
                with k_col3:
                    r0c2 = st.number_input("K[0,2]", step=1.0, key="m_02")
                    r1c2 = st.number_input("K[1,2]", step=1.0, key="m_12")
                    r2c2 = st.number_input("K[2,2]", step=1.0, key="m_22")

                kernel_matrix = np.array(
                    [[r0c0, r0c1, r0c2], [r1c0, r1c1, r1c2], [r2c0, r2c1, r2c2]],
                    dtype=np.float32,
                )

                normalize_flag = st.checkbox(
                    "啟用自動正規化 (Normalization)", key="norm_chk", help="將卷積核權重總和正規化為 1，避免亮度過曝"
                )

                if normalize_flag:
                    k_sum = np.sum(kernel_matrix)
                    if k_sum != 0:
                        kernel_matrix = kernel_matrix / k_sum

                filtered_bgr = cv2.filter2D(img_bgr, -1, kernel_matrix)
                filtered_bgr = np.clip(filtered_bgr, 0, 255).astype(np.uint8)
                filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)

                mse = np.mean((img_bgr.astype(np.float64) - filtered_bgr.astype(np.float64)) ** 2)
                if mse == 0:
                    psnr_val = float('inf')
                else:
                    psnr_val = 20 * math.log10(255.0 / math.sqrt(mse))

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.image(img_rgb, caption="原始影像", use_container_width=True)
                with col_f2:
                    st.image(filtered_rgb, caption="濾波處理後影像", use_container_width=True)

                st.markdown("---")
                if psnr_val == float('inf'):
                    st.metric("📊 峰值信噪比 (PSNR)", "∞ dB", delta="完全無失真")
                else:
                    st.metric("📊 峰值信噪比 (PSNR)", f"{psnr_val:.2f} dB")

                with st.expander("📐 查看 PSNR 計算公式與說明"):
                    st.latex(r"MSE = \frac{1}{3 m n} \sum_{c=1}^{3} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j,c) - K(i,j,c)]^2")
                    st.latex(r"PSNR = 10 \cdot \log_{10} \left( \frac{255^2}{MSE} \right) = 20 \cdot \log_{10} \left( \frac{255}{\sqrt{MSE}} \right)")

            # --- Tab 4: AI 人臉偵測模組 (OpenCV ONNX YuNet 深度學習) ---
            with tab4:
                st.subheader("OpenCV ONNX YuNet AI 人臉偵測模組")

                conf_threshold = 0.50
                annotated_img = img_rgb.copy()
                face_count = 0
                face_details = []

                try:
                    model_path = get_yunet_onnx_model_path()
                    detector = cv2.FaceDetectorYN.create(
                        model=model_path,
                        config="",
                        input_size=(int(width), int(height)),
                        score_threshold=float(conf_threshold),
                        nms_threshold=0.3,
                        top_k=500
                    )

                    _, faces = detector.detect(img_bgr)

                    if faces is not None:
                        face_count = len(faces)
                        for idx, face in enumerate(faces):
                            box = list(map(int, face[0:4]))
                            confidence = float(face[-1])

                            xmin, ymin, bw, bh = box
                            xmin, ymin = max(0, xmin), max(0, ymin)

                            cv2.rectangle(annotated_img, (xmin, ymin), (xmin + bw, ymin + bh), (0, 255, 0), 3)
                            label = f"Face #{idx+1}: {confidence*100:.1f}%"
                            cv2.putText(annotated_img, label, (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            face_details.append({
                                "人臉編號": idx + 1,
                                "置信度 (Confidence)": f"{confidence*100:.2f}%",
                                "邊界框 (x, y, w, h)": f"({xmin}, {ymin}, {bw}, {bh})"
                            })

                    st.caption("🤖 偵測技術：OpenCV YuNet Deep Learning Face Detector (ONNX Standard)")

                except Exception as e:
                    st.error(f"⚠️ 載入或執行 ONNX 模型時發生錯誤：{e}")

                col_res1, col_res2 = st.columns([1.3, 0.7])
                with col_res1:
                    st.image(
                        annotated_img,
                        caption="人臉偵測結果標註圖",
                        use_container_width=True,
                    )

                with col_res2:
                    st.markdown("#### 📋 偵測結果摘要")
                    st.metric(label="偵測到的總人臉數量", value=f"{face_count} 人")
                    st.markdown("---")
                    if face_count > 0:
                        st.dataframe(face_details, use_container_width=True)
                    else:
                        st.warning("未偵測到人臉。")

        else:
            st.error("❌ 影像解碼失敗，請確認檔案格式是否正確。")

    except Exception as e:
        st.error(f"❌ 讀取過程發生異常錯誤: {e}")

else:
    st.info("👈 請由左側側邊欄上傳支援的點陣圖檔案（PNG, JPG, JPEG, BMP）以啟動系統。")
