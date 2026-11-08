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

st.title("🖼️ AI 輔助影像處理與人臉分析系統")
st.write(
    "期中專題實作介面 —— 整合色彩空間分析、手動卷積濾波與智慧人臉檢測"
)

# -------------------------------------------------------------
# 核心一：影像載入與基礎解析
# -------------------------------------------------------------
st.sidebar.header("📁 影像檔案上傳")
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
          "1. 原始與中繼",
          "2. 色彩空間與直方圖",
          "3. 卷積濾波與 PSNR",
          "4. 智慧人臉偵測",
      ])

      # --- Tab 1: 原始影像與預覽 ---
      with tab1:
        st.subheader("原始影像即時預覽")
        st.image(
            img_rgb,
            caption=f"Original Image: {uploaded_file.name}",
            channels="RGB",
            use_container_width=True,
        )
        st.info("💡 提示：您可以透過左側選單隨時更換不同照片。")

      # --- Tab 2: 多色彩空間轉換與同步直方圖分析 ---
      with tab2:
        st.subheader("🎨 核心二：多色彩空間與同步直方圖分析")
        st.write(
            "系統自動將影像轉換至不同色彩空間與通道，並同步繪製對應的像素強度直方圖。"
        )

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(
            ["RGB 三通道檢視", "單通道顏色分離", "YUV 色彩空間"]
        )

        def plot_histogram(img_channel, title, color):
          fig, ax = plt.subplots(figsize=(6, 2.8))
          hist = cv2.calcHist([img_channel], [0], None, [256], [0, 256])
          ax.plot(hist, color=color, linewidth=1.5)
          ax.fill_between(
              range(256), hist.flatten(), color=color, alpha=0.2
          )
          ax.set_title(title, fontsize=11)
          ax.set_xlabel("Pixel Intensity (0-255)", fontsize=9)
          ax.set_ylabel("Count", fontsize=9)
          ax.set_xlim([0, 256])
          ax.grid(True, linestyle="--", alpha=0.4)
          st.pyplot(fig)
          plt.close(fig)

        with sub_tab1:
          col1, col2 = st.columns(2)
          with col1:
            st.image(img_rgb, caption="RGB 原圖", use_container_width=True)
          with col2:
            st.markdown("#### RGB 三通道強度分布")
            fig, ax = plt.subplots(figsize=(6, 2.8))
            colors = ("red", "green", "blue")
            for i, col in enumerate(colors):
              hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
              ax.plot(hist, color=col, label=f"{col.upper()} 頻道")
            ax.set_xlim([0, 256])
            ax.set_xlabel("Pixel Intensity (0-255)")
            ax.set_ylabel("Count")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig)
            plt.close(fig)

        with sub_tab2:
          channel_choice = st.selectbox(
              "選擇要隔離的色彩通道",
              ["紅色通道 (R)", "綠色通道 (G)", "藍色通道 (B)"],
              key="ch_select",
          )
          idx = 0
          c_name, c_color = "Red", "red"
          if channel_choice == "綠色通道 (G)":
            idx, c_name, c_color = 1, "Green", "green"
          elif channel_choice == "藍色通道 (B)":
            idx, c_name, c_color = 2, "Blue", "blue"

          single_channel = np.zeros_like(img_rgb)
          single_channel[:, :, idx] = img_rgb[:, :, idx]

          col1, col2 = st.columns(2)
          with col1:
            st.image(
                single_channel,
                caption=f"{channel_choice} 隔離影像",
                use_container_width=True,
            )
          with col2:
            st.markdown(f"#### {c_name} 通道直方圖")
            plot_histogram(
                img_rgb[:, :, idx], f"{c_name} Channel Histogram", c_color
            )

        with sub_tab3:
          img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
          y_ch, u_ch, v_ch = cv2.split(img_yuv)

          yuv_choice = st.radio(
              "選擇 YUV 分量模式",
              ["Y (亮度)", "U (色度)", "V (色度)"],
              horizontal=True,
              key="yuv_radio",
          )

          col1, col2 = st.columns(2)
          if yuv_choice == "Y (亮度)":
            with col1:
              st.image(
                  y_ch,
                  caption="Y 亮度分量",
                  use_container_width=True,
              )
            with col2:
              st.markdown("#### Y 分量直方圖")
              plot_histogram(y_ch, "Y Channel Histogram", "gray")
          elif yuv_choice == "U (色度)":
            with col1:
              st.image(
                  u_ch,
                  caption="U 色度分量",
                  use_container_width=True,
              )
            with col2:
              st.markdown("#### U 分量直方圖")
              plot_histogram(u_ch, "U Channel Histogram", "blue")
          else:
            with col1:
              st.image(
                  v_ch,
                  caption="V 色度分量",
                  use_container_width=True,
              )
            with col2:
              st.markdown("#### V 分量直方圖")
              plot_histogram(v_ch, "V Channel Histogram", "red")

      # --- Tab 3: 核心三 - 手動矩陣與 PSNR 評估 ---
      with tab3:
        st.subheader("⚙️ 核心三：自訂卷積核矩陣與 PSNR 評估")
        st.write(
            "提供 3×3 卷積核矩陣的編輯功能。切換上方範本時將自動連動更新下方矩陣數值，"
            "您也可以隨時手動微調，系統將即時計算 PSNR。"
        )

        # 初始化 session_state，確保所有 key 都有預設值
        if "m_00" not in st.session_state: st.session_state.m_00 = 0.0
        if "m_01" not in st.session_state: st.session_state.m_01 = 0.0
        if "m_02" not in st.session_state: st.session_state.m_02 = 0.0
        if "m_10" not in st.session_state: st.session_state.m_10 = 0.0
        if "m_11" not in st.session_state: st.session_state.m_11 = 1.0
        if "m_12" not in st.session_state: st.session_state.m_12 = 0.0
        if "m_20" not in st.session_state: st.session_state.m_20 = 0.0
        if "m_21" not in st.session_state: st.session_state.m_21 = 0.0
        if "m_22" not in st.session_state: st.session_state.m_22 = 0.0
        if "norm_chk" not in st.session_state: st.session_state.norm_chk = False

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
          elif preset == "盒狀模糊 (Box Blur)":
            vals = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
            st.session_state.norm_chk = True
          else: # 手動自訂
            return

          st.session_state.m_00 = float(vals[0][0])
          st.session_state.m_01 = float(vals[0][1])
          st.session_state.m_02 = float(vals[0][2])
          st.session_state.m_10 = float(vals[1][0])
          st.session_state.m_11 = float(vals[1][1])
          st.session_state.m_12 = float(vals[1][2])
          st.session_state.m_20 = float(vals[2][0])
          st.session_state.m_21 = float(vals[2][1])
          st.session_state.m_22 = float(vals[2][2])

        st.selectbox(
            "快速濾波範本 (選擇後將即時連動下方 9 個輸入框)",
            [
                "手動自訂矩陣",
                "高斯平滑 (Gaussian Blur)",
                "拉普拉斯銳化 (Sharpen)",
                "邊緣檢測 (Edge Detection)",
                "盒狀模糊 (Box Blur)",
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
            "啟用自動正規化 (Normalization)", key="norm_chk"
        )

        if normalize_flag:
          k_sum = np.sum(kernel_matrix)
          if k_sum != 0:
            kernel_matrix = kernel_matrix / k_sum

        filtered_bgr = cv2.filter2D(img_bgr, -1, kernel_matrix)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        psnr_value = cv2.PSNR(img_bgr, filtered_bgr)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
          st.image(img_rgb, caption="原始影像", use_container_width=True)
        with col_f2:
          st.image(
              filtered_rgb, caption="濾波處理後影像", use_container_width=True
          )

        st.markdown("---")
        st.metric(
            label="📈 峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)",
            value=f"{psnr_value:.2f} dB",
        )

      # --- Tab 4: 智慧人臉偵測模組 ---
      with tab4:
        st.subheader("👤 核心四：智慧人臉自動偵測與統計")
        st.write(
            "系統採用標準 Haar Cascade 分類器進行特徵檢測，"
            "並過濾異常的候選區塊，確保標記框皆為真實人臉。"
        )

        img_detected = img_bgr.copy()
        faces = []

        try:
          if hasattr(cv2, "CascadeClassifier"):
            xml_name = "haarcascade_frontalface_default.xml"
            if not os.path.exists(xml_name):
              try:
                url = f
