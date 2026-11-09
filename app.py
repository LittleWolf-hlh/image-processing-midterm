@st.cache_resource
def get_yunet_onnx_model_path() -> str:
    """下載並快取 OpenCV 官方 YuNet ONNX 人臉偵測模型，並驗證檔案正確性"""
    onnx_path = "face_detection_yunet_2023mar.onnx"
    onnx_url = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    # 如果檔案不存在，或檔案內容包含 HTML（代表下載失敗變成 404 頁面），則重新下載
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
