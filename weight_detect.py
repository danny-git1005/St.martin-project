import sys
import pandas as pd
import os
import cv2
import onnxruntime as ort
import logging
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QProgressBar, QAbstractItemView,
    QDialog, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor
import configparser
# 假設 process_video 函數在 tool.utils 模組中
from tool.utils import process_video

# Setup logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessingWorker(QThread):
    progress_update = pyqtSignal(int, str, str, pd.DataFrame)

    def __init__(self, video_files, crop_img):
        super().__init__()
        self.video_files = video_files
        self.crop_img = crop_img  # [X, Y, W, H]

    def run(self):
        total_videos = len(self.video_files)
        for i, video_file in enumerate(self.video_files):
            filename = os.path.basename(video_file)
            logging.info(f"Processing video: {filename}")
            logging.info(f"Crop coordinates: {self.crop_img}")
            
            # 呼叫處理影片的函數，並傳入 crop_img
            measure_df = process_video(video_file, self.crop_img)
            csv_file = os.path.join(".", f"{filename}_result.csv")

            # 發送進度更新信號，包含 measure_df
            progress = int(((i + 1) / total_videos) * 100)
            self.progress_update.emit(progress, video_file, csv_file, measure_df)

class VideoProcessingApp(QWidget):
    def __init__(self):
        super().__init__()

        # 這裡添加變數來存儲框選範圍
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = 0
        self.crop_h = 0

        self.setWindowTitle("重量偵測程式")
        self.setGeometry(100, 100, 800, 700)  # 調整視窗大小

        # 主佈局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # GPU 狀態按鈕 (啟動程式時自動檢查)
        # self.gpu_button = QPushButton("正在檢查 GPU...")
        # self.gpu_button.setFixedSize(150, 40)
        # self.check_gpu_status()  # 啟動時自動檢查 GPU 狀態
        # self.layout.addWidget(self.gpu_button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # 顯示已上傳的影片數量的標籤
        self.label = QLabel("尚未上傳影片")
        self.layout.addWidget(self.label)

        # 影片列表（垂直顯示）
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.layout.addWidget(self.video_list)

        # 底部的佈局來放置上傳、選擇全部按鈕
        self.button_layout = QHBoxLayout()

        # 上傳影片的按鈕
        self.upload_button = QPushButton("上傳影片")
        self.upload_button.setFixedSize(100, 30)
        self.upload_button.clicked.connect(self.upload_videos)
        self.button_layout.addWidget(self.upload_button)

        # 選擇全部影片的按鈕
        self.select_all_button = QPushButton("選擇全部")
        self.select_all_button.setFixedSize(100, 30)
        self.select_all_button.clicked.connect(self.select_all_videos)
        self.button_layout.addWidget(self.select_all_button)

        self.layout.addLayout(self.button_layout)

        # 新增 "顯示圖片" 的按鈕
        self.show_image_button = QPushButton("顯示框選範圍")
        self.show_image_button.setFixedSize(100, 30)
        self.show_image_button.clicked.connect(self.display_image)
        self.layout.addWidget(self.show_image_button)

        # 顯示和調整框選範圍的 UI
        self.crop_layout = QHBoxLayout()

        self.crop_x_label = QLabel("橫軸:")
        self.crop_x_value = QLabel("880")
        self.crop_layout.addWidget(self.crop_x_label)
        self.crop_layout.addWidget(self.crop_x_value)

        self.crop_y_label = QLabel("縱軸:")
        self.crop_y_value = QLabel("240")
        self.crop_layout.addWidget(self.crop_y_label)
        self.crop_layout.addWidget(self.crop_y_value)

        self.crop_w_label = QLabel("寬:")
        self.crop_w_value = QLabel("120")
        self.crop_layout.addWidget(self.crop_w_label)
        self.crop_layout.addWidget(self.crop_w_value)

        self.crop_h_label = QLabel("高:")
        self.crop_h_value = QLabel("60")
        self.crop_layout.addWidget(self.crop_h_label)
        self.crop_layout.addWidget(self.crop_h_value)

        self.layout.addLayout(self.crop_layout)

        # 處理進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        # 開始處理影片的按鈕
        self.process_button = QPushButton("開始處理")
        self.process_button.clicked.connect(self.process_videos)
        self.layout.addWidget(self.process_button)

        # 處理完成的 CSV 檔案列表
        self.csv_list = QListWidget()
        self.csv_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.layout.addWidget(self.csv_list)

        # 選擇全部 CSV 檔案的按鈕
        self.select_all_csv_button = QPushButton("選擇全部 CSV")
        self.select_all_csv_button.setFixedSize(150, 30)
        self.select_all_csv_button.clicked.connect(self.select_all_csv)
        self.layout.addWidget(self.select_all_csv_button)

        # 下載 CSV 檔案的按鈕
        self.download_csv_button = QPushButton("下載 CSV 檔案")
        self.download_csv_button.setFixedSize(150, 30)
        self.download_csv_button.clicked.connect(self.download_selected_csv)
        self.layout.addWidget(self.download_csv_button)

        # 用來存儲影片路徑的暫存區
        self.video_files = []
        self.processed_files = []  # 用來存儲處理過的 CSV 檔案
        self.measured_dfs = {}  # 存儲每個影片的測量結果 DataFrame

         # 加載配置文件
        self.config = configparser.ConfigParser()
        self.load_config()

        # 初始框選範圍
        self.crop_img = [self.config.getint('CROP', 'X'),
                         self.config.getint('CROP', 'Y'),
                         self.config.getint('CROP', 'W'),
                         self.config.getint('CROP', 'H')]

        # 其餘代碼...
    
    def load_config(self):
        config_file = "settings.ini"
        if os.path.exists(config_file):
            self.config.read(config_file)
        else:
            # 如果配置文件不存在，則使用默認值並創建配置文件
            self.config['CROP'] = {'X': '880', 'Y': '240', 'W': '120', 'H': '60'}
            with open(config_file, 'w') as configfile:
                self.config.write(configfile)

    def update_crop_labels(self):
        logging.info(f"Updating crop labels to: {self.crop_img}")
        self.crop_x_value.setText(str(self.crop_img[0]))
        self.crop_y_value.setText(str(self.crop_img[1]))
        self.crop_w_value.setText(str(self.crop_img[2]))
        self.crop_h_value.setText(str(self.crop_img[3]))

    def update_crop_values(self, x, y, w, h):
        self.crop_img = [x, y, w, h]
        logging.info(f"Updating crop values to: {self.crop_img}")
        self.update_crop_labels()

        # 更新配置文件
        self.config['CROP']['X'] = str(x)
        self.config['CROP']['Y'] = str(y)
        self.config['CROP']['W'] = str(w)
        self.config['CROP']['H'] = str(h)
        with open("settings.ini", 'w') as configfile:
            self.config.write(configfile)

    def upload_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "上傳影片", "", "Videos (*.mp4 *.avi *.mov *.dav)")
        if files:
            self.video_files = files
            self.video_list.clear()
            for file in files:
                self.video_list.addItem(file)
            self.label.setText(f"已上傳 {len(files)} 部影片")

            # 載入第一個影片的第一幀
            first_frame = self.load_first_frame(files[0])
            if first_frame is not None:
                self.first_frame = first_frame
                self.update_crop_labels()
            else:
                QMessageBox.warning(self, "載入失敗", "無法讀取第一個影片的第一幀。")

    def load_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            logging.info("影片第一幀已成功載入。")
            return frame
        else:
            logging.error("無法讀取影片的第一幀。")
            return None

    def select_all_videos(self):
        for index in range(self.video_list.count()):
            item = self.video_list.item(index)
            item.setSelected(True)

    def select_all_csv(self):
        for index in range(self.csv_list.count()):
            item = self.csv_list.item(index)
            item.setSelected(True)

    def display_image(self):
        if not self.video_files:
            QMessageBox.warning(self, "警告", "請先上傳影片。")
            return
        if not hasattr(self, 'first_frame') or self.first_frame is None:
            QMessageBox.warning(self, "警告", "無法載入影片的第一幀。")
            return

        dialog = ImageCropperDialog(self.first_frame, self.crop_img, self)
        dialog.crop_selected.connect(self.update_crop_values)
        
        dialog.exec()

    def process_videos(self):
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            self.label.setText("未選擇影片")
            return

        self.progress_bar.setValue(0)
        self.processed_files.clear()
        self.measured_dfs.clear()

        self.worker = VideoProcessingWorker([item.text() for item in selected_items], self.crop_img)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def update_progress(self, progress, video_file, csv_file, measure_df):
        logging.info(f"Video processing progress: {progress}% for file: {video_file}")
        self.csv_list.addItem(csv_file)
        self.processed_files.append(csv_file)
        self.measured_dfs[csv_file] = measure_df
        self.progress_bar.setValue(progress)

    def on_processing_finished(self):
        logging.info("影片處理完成！")
        self.label.setText("影片處理完成！")

    def download_selected_csv(self):
        selected_items = self.csv_list.selectedItems()
        if not selected_items:
            self.label.setText("未選擇要下載的 CSV 檔案")
            return

        save_directory = QFileDialog.getExistingDirectory(self, "選擇下載位置")
        if save_directory:
            for item in selected_items:
                csv_file_path = item.text()
                if csv_file_path in self.measured_dfs:
                    destination_path = os.path.join(save_directory, os.path.basename(csv_file_path))
                    measure_df = self.measured_dfs[csv_file_path]
                    measure_df.to_csv(destination_path, index=False, encoding='utf-8-sig')
            logging.info(f"CSV 檔案已下載到 {save_directory}")
            self.label.setText(f"CSV 檔案已下載到 {save_directory}")

    def check_gpu_status(self):
        device = ort.get_device()
        logging.info(f"Detected device: {device}")
        if device == "GPU":
            self.gpu_button.setStyleSheet("background-color: green; color: white;")
            self.gpu_button.setText("GPU 可用")
            logging.info("GPU is available")
        else:
            self.gpu_button.setStyleSheet("background-color: orange; color: white;")
            self.gpu_button.setText("僅有 CPU 可用")
            logging.warning("Only CPU is available")

class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, x, y, w, h, img_h, img_w, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRect(x, y, w, h)
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.resizing = False  # 標記是否正在調整大小
        self.resize_handle_size = 8  # 調整手柄的大小
        self.img_h = img_h  # 圖片高度
        self.img_w = img_w  # 圖片寬度

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        # 畫出調整手柄
        if self.isSelected():
            rect = self.rect()
            painter.setBrush(QColor(0, 0, 255))  # 藍色手柄
            painter.drawRect(int(rect.x() + rect.width() - self.resize_handle_size),
                             int(rect.y() + rect.height() - self.resize_handle_size),
                             int(self.resize_handle_size),
                             int(self.resize_handle_size))

    def mousePressEvent(self, event):
        # 檢查是否在手柄上按下
        if (event.pos().x() >= self.rect().x() + self.rect().width() - self.resize_handle_size and
            event.pos().y() >= self.rect().y() + self.rect().height() - self.resize_handle_size):
            self.resizing = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            # 計算新的矩形框大小和位置
            new_x = self.rect().x()
            new_y = self.rect().y()
            new_width = event.pos().x() - self.rect().x()
            new_height = event.pos().y() - self.rect().y()

            # 邊界檢查，確保矩形框不超出圖片邊界
            if new_x + new_width > self.img_w:
                new_width = self.img_w - new_x
            if new_y + new_height > self.img_h:
                new_height = self.img_h - new_y

            # 確保寬高不能小於最小手柄大小
            if new_width < self.resize_handle_size:
                new_width = self.resize_handle_size
            if new_height < self.resize_handle_size:
                new_height = self.resize_handle_size

            # 更新矩形框大小
            self.setRect(new_x, new_y, new_width, new_height)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.resizing:
            self.resizing = False

        # 获取矩形框的场景坐标
        scene_position = self.scenePos()
        
        # 计算相对于图片的坐标
        x = int(scene_position.x() + self.rect().x())
        y = int(scene_position.y() + self.rect().y())
        w = int(self.rect().width())
        h = int(self.rect().height())

        logging.info(f"更新位置: ({x}, {y}), 尺寸: ({w}, {h})")

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + w > self.img_w:
            x = self.img_w - w
        if y + h > self.img_h:
            y = self.img_h - h

        super().mouseReleaseEvent(event)

class ImageCropperDialog(QDialog):
    crop_selected = pyqtSignal(int, int, int, int)  # X, Y, W, H

    def __init__(self, image, initial_crop, parent=None):
        super().__init__(parent)
        self.setWindowTitle("調整框選範圍")
        self.setGeometry(150, 150, 1920, 1080)

        self.image = image
        self.ori_img_h, self.ori_img_w, self.ori_img_c = image.shape
        self.initial_crop = initial_crop  # [X, Y, W, H]

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 使用 QGraphicsView 來顯示圖片和框選範圍
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.layout.addWidget(self.graphics_view)

        # 將影像轉換為 QImage 並添加到場景中
        self.qimage = self.convert_cv_qt(self.image)
        self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(self.qimage))

        # 添加可調整的矩形框
        x, y, w, h = self.initial_crop
        self.rect_item = ResizableRectItem(x, y, w, h, self.ori_img_h, self.ori_img_w)  # 使用新的 ResizableRectItem
        self.rect_item.setPen(QPen(QColor(255, 0, 0), 2))   # 紅色邊框
        self.rect_item.setBrush(QColor(255, 0, 0, 100))     # 可視化填充
        self.scene.addItem(self.rect_item)

        # 確認和取消按鈕
        self.button_layout = QHBoxLayout()
        self.ok_button = QPushButton("確認")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return q_image

    def accept(self):
        # 獲取矩形框的位置和大小
        rect = self.rect_item.rect()
        scene_position = self.rect_item.scenePos()
        
        x = int(scene_position.x() + rect.x())
        y = int(scene_position.y() + rect.y())
        w = int(rect.width())
        h = int(rect.height())
        
        logging.info(f"Crop confirmed: x={x}, y={y}, w={w}, h={h}")
        
        self.crop_selected.emit(x, y, w, h)
        super().accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())
