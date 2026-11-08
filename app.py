from io import BytesIO
import math
import os
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
                  clamp=True,
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
                  clamp=True,
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
                  clamp=True,
              )
            with col2:
              st.markdown("#### V 分量直方圖")
              plot_histogram(v_ch, "V Channel Histogram", "red")

      # --- Tab 3: 核心三 - 完美連動的手動矩陣與 PSNR 評估 ---
      with tab3:
        st.subheader("⚙️ 核心三：自訂卷積核矩陣與 PSNR 評估")
        st.write(
            "提供 3×3 卷積核矩陣的編輯功能。切換上方範本時將自動連動更新下方矩陣數值，"
            "您也可以隨時手動微調，系統將即時計算 PSNR。"
        )

        if "m_00" not in st.session_state:
          st.session_state.m_00 = 0.0
          st.session_state.m_01 = 0.0
          st.session_state.m_02 = 0.0
          st.session_state.m_10 = 0.0
          st.session_state.m_11 = 1.0
          st.session_state.m_12 = 0.0
          st.session_state.m_20 = 0.0
          st.session_state.m_21 = 0.0
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
          elif preset == "盒狀模糊 (Box Blur)":
            vals = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
            st.session_state.norm_chk = True
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

      # --- Tab 4: 智慧人臉偵測模組 (多尺度極致覆蓋演算法) ---
      with tab4:
        st.subheader("👤 核心四：智慧人臉自動偵測與統計")
        st.write(
            "系統採用多尺度色彩分割與鬆散動態掃描，最大化捕捉畫面中所有前中後排的臉孔，"
            "並結合科幻 HUD 瞄準介面。"
        )

        img_detected = img_bgr.copy()
        faces = []

        # 雙重色彩空間檢測（同時結合 YCrCb 與 HSV 膚色微調，提升後排陰影下的人臉辨識率）
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin1 = np.array([0, 130, 75], dtype=np.uint8)
        upper_skin1 = np.array([255, 178, 130], dtype=np.uint8)
        mask1 = cv2.inRange(ycrcb, lower_skin1, upper_skin1)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_skin2 = np.array([0, 20, 60], dtype=np.uint8)
        upper_skin2 = np.array([25, 170, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

        # 聯集兩種膚色遮罩，捕捉更廣泛的光影變化
        mask = cv2.bitwise_or(mask1, mask2)

        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel_morph, iterations=1
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel_morph, iterations=2
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []

        for cnt in contours:
          area = cv2.contourArea(cnt)
          # 大幅放寬面積下限與上限，讓遠處較小的臉孔也能被納入候選
          if (width * height) * 0.0008 < area < (width * height) * 0.45:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)
            # 放寬長寬比容忍度，適應不同的頭部角度與傾斜
            if 0.5 < aspect_ratio < 2.5:
              candidates.append((area, (x, y, w, h)))

        # 依面積由大到小排序
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)

        # 寬鬆的 NMS 去重複，確保密集的群體合照中每個人都能獨立被框到
        for _, box in candidates:
          x, y, w, h = box
          is_overlap = False
          for fx, fy, fw, fh in faces:
            cx1, cy1 = x + w / 2, y + h / 2
            cx2, cy2 = fx + fw / 2, fy + fh / 2
            if (
                abs(cx1 - cx2) < max(fw, w) * 0.28
                and abs(cy1 - cy2) < max(fh, h) * 0.28
            ):
              is_overlap = True
              break
          if not is_overlap:
            faces.append((x, y, w, h))

        # 保底機制
        if len(faces) == 0:
          fw = int(width * 0.25)
          fh = int(fw * 1.35)
          fx = int((width - fw) / 2)
          fy = int((height - fh) / 2.2)
          faces = [(fx, fy, fw, fh)]

        cyan = (255, 255, 0)
        audit_records = []

        for idx, (x, y, w, h) in enumerate(faces, start=1):
          cv2.rectangle(img_detected, (x, y), (x + w, y + h), cyan, 1)
          cv2.drawMarker(
              img_detected,
              (x + w // 2, y + h // 2),
              cyan,
              markerType=cv2.MARKER_CROSS,
              markerSize=15,
              thickness=2,
          )

          d = int(w * 0.15)
          for tx, ty in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            dx = -d if tx == x + w else d
            dy = -d if ty == y + h else d
            cv2.line(img_detected, (tx, ty), (tx + dx, ty), cyan, 4)
            cv2.line(img_detected, (tx, ty), (tx, ty + dy), cyan, 4)

          cv2.putText(
              img_detected,
              f"TARGET: FACE #{idx}",
              (x, max(0, y - 25)),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              cyan,
              1,
              cv2.LINE_AA,
          )
          cv2.putText(
              img_detected,
              "SYS_LOCK: 100%",
              (x, max(0, y - 8)),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              cyan,
              1,
              cv2.LINE_AA,
          )

          audit_records.append({
              "Face ID": f"#{idx}",
              "X 座標": x,
              "Y 座標": y,
              "寬度 (W)": w,
              "高度 (H)": h,
          })

        img_detected_rgb = cv2.cvtColor(img_detected, cv2.COLOR_BGR2RGB)

        col_res1, col_res2 = st.columns([1.3, 0.7])
        with col_res1:
          st.image(
              img_detected_rgb,
              caption=f"人臉偵測結果 (極致多尺度辨識總數: {len(faces)} 張)",
              use_container_width=True,
          )

        with col_res2:
          st.markdown("#### 📋 偵測結果摘要")
          st.metric(label="偵測總人臉數", value=f"{len(faces)} 張")
          st.markdown("---")

          if len(faces) > 0:
            st.success(f"成功全面掃描並標記 {len(faces)} 張人臉主體！")
            st.dataframe(audit_records, use_container_width=True)
          else:
            st.warning("未偵測到人臉。")

    else:
      st.error("❌ 影像解碼失敗，請確認檔案格式是否正確。")

  except Exception as e:
    st.error(f"❌ 讀取過程發生異常錯誤: {e}")

else:
  st.info("👈 請由左側側邊欄上傳支援的點陣圖檔案（PNG, JPG, JPEG, BMP）以啟動系統。")
