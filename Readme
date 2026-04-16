import os

readme_content = """# 🤟 ASL Sign Language Detection System (YOLOv11)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-red.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV-green.svg)](https://opencv.org/)
[![mAP](https://img.shields.io/badge/mAP%4050-96.67%25-brightgreen.svg)]()

Hệ thống nhận diện Ngôn ngữ Ký hiệu Mỹ (ASL) thời gian thực sử dụng kiến trúc **YOLOv11**. Dự án được thiết kế để hỗ trợ học tập và giao tiếp cho người khiếm thính thông qua các tính năng tương tác thông minh.

---

## ✨ Tính năng nổi bật

* **Nhận diện Real-time:** Tối ưu hóa đạt tốc độ xử lý cực nhanh (~90 FPS), đảm bảo trải nghiệm mượt mà.
* **Độ chính xác cao:** Model đạt chỉ số **96.67% mAP@50** và **90.02% mAP@50-95**.
* **Temporal Smoothing:** Thuật toán trượt cửa sổ (Sliding Window Voting) giúp loại bỏ hiện tượng nhấp nháy nhãn (flickering).
* **Word Builder Mode:** Ghép các chữ cái đã nhận diện thành từ hoàn chỉnh, hỗ trợ xóa và lưu lịch sử.
* **Interactive Quiz Mode:** Chế độ luyện tập 3 mức độ (Easy, Normal, Hard) yêu cầu người dùng giữ đúng ký hiệu dưới áp lực thời gian.
* **History Logging:** Tự động lưu trữ lịch sử từ vựng vào file `word_history.json`.

---

## 🛠 Công nghệ sử dụng

* **Deep Learning:** YOLOv11 (Ultralytics), PyTorch.
* **Computer Vision:** OpenCV.
* **Data Management:** Roboflow (Preprocessing & Augmentation).
* **Environment:** Google Colab (Huấn luyện với GPU NVIDIA H100).
* **Storage:** JSON-based logging.

---

## 📊 Thông số kỹ thuật

| Metric | Giá trị |
| :--- | :--- |
| **Số lượng ảnh gốc** | 10,045 images |
| **mAP@50** | **96.67%** |
| **mAP@50-95** | **90.02%** |
| **Precision** | **93.80%** |
| **Recall** | **95.65%** |
| **Inference Speed** | **11.1ms / frame** |

---

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường
```bash
pip install ultralytics opencv-python torch torchvision