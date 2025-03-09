import cv2
import pandas as pd
import time
import logging
from .yolov6_utils import yolov6

# Setup logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_image(image, x, y, w, h):
    """裁剪图像并更新 YOLO 标签"""
    logging.info(f"Cropping image at coordinates: x={x}, y={y}, w={w}, h={h}")
    return image[y:y+h, x:x+w]

def convert_to_number(data):
    sorted_data = sorted(data, key=lambda x: x[0])
    length = len(sorted_data)
    number = 0
    for i in range(length):
        number += sorted_data[i][-1]
        if i == length - 1:
            number /= 10
        else:
            number *= 10

    logging.info(f"Converted prediction data to number: {number}")
    return number

def process_frame(frame, model_local, crop_xywh):
    crop_x, crop_y, crop_w, crop_h = crop_xywh
    # logging.info(f"Processing frame with crop coordinates: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    processed_frame = crop_image(frame, crop_x, crop_y, crop_w, crop_h)
    processed_frame = cv2.resize(processed_frame, (240, 120))
    _, prediction = model_local.detect(processed_frame)

    number = 0
    if len(prediction) > 0:
        number = convert_to_number(prediction)
        if number <= 1.0:
            number = 0.0
    else:
        logging.error(f"convert : error, {prediction}")
        number = -1

    logging.info(f"Frame processed, detected number: {number}")
    return number

def parse_time_string(time_string):
    year = int(time_string[:4])
    month = int(time_string[4:6])
    day = int(time_string[6:8])
    hour = int(time_string[8:10])
    minute = int(time_string[10:12])
    second = int(time_string[12:14])
    return year, month, day, hour, minute, second

def increment_time(year, month, day, hour, minute, second):
    second += 1
    if second >= 60:
        second = 0
        minute += 1
    if minute >= 60:
        minute = 0
        hour += 1
    if hour >= 24:
        hour = 0
        day += 1
    if day > 31:
        day = 1
        month += 1
    if month > 12:
        month = 1
        year += 1
    return year, month, day, hour, minute, second

def time2str(year, month, day, hour, minute, second):
    str_month = f"0{month}" if month < 10 else str(month)
    str_day = f"0{day}" if day < 10 else str(day)
    str_hour = f"0{hour}" if hour < 10 else str(hour)
    str_minute = f"0{minute}" if minute < 10 else str(minute)
    str_second = f"0{second}" if second < 10 else str(second)
    return f"{year}-{str_month}-{str_day}-{str_hour}:{str_minute}:{str_second}"

def analyze_number_date(df):
    result = []
    current_number = None
    count = 0
    for i, row in df.iterrows():
        if row['number'] == current_number:
            count += 1
        else:
            if current_number is not None:
                result.append([current_number, count, previous_date])
            current_number = row['number']
            count = 1
        previous_date = row['date']
    result.append([current_number, count, previous_date])
    result_df = pd.DataFrame(result, columns=['number', 'count', 'date'])
    condition = (result_df['number'] == 0.0) & (result_df['count'] > 30)
    result_df['group'] = condition.cumsum()
    subsequences = [sub_df for _, sub_df in result_df.groupby('group') if not (sub_df['number'].nunique() == 1 and sub_df['number'].iloc[0] == 0.0)]
    results = []
    measure_count = 1
    for i, subsequence in enumerate(subsequences):
        grouped_df = subsequence.groupby(['number', 'group']).agg(
            count=('count', 'sum'),
            date=('date', 'max')
        ).reset_index()
        filtered_subseq = grouped_df[(grouped_df['number'] >= 1.0) & (grouped_df['count'] > 30)]
        if not filtered_subseq.empty:
            max_row = filtered_subseq.loc[filtered_subseq['count'].idxmax()]
            results.append((measure_count, max_row['number'], max_row['date']))
            measure_count += 1
    df_max_values = pd.DataFrame(results, columns=["測量", "數值", "時間"])
    return df_max_values

def merge_duplicates(sequence):
    merged_sequence = {}
    for value, count in sequence:
        if value in merged_sequence:
            merged_sequence[value] += count
        else:
            merged_sequence[value] = count
    return list(merged_sequence.items())

def process_video(filepath, crop_xywh):
    logging.info(f"Starting video processing for file: {filepath}")
    cap = cv2.VideoCapture(filepath)
    model_local = yolov6(
        "./onnx_model/yolov6s.onnx",
        confThreshold=0.7,
        nmsThreshold=0.5
    )

    data = []
    frame_count = 1
    filename = filepath.split("\\")[-1]
    date = filename.split("_")[3]
    year, month, day, hour, minute, second = parse_time_string(date)
    no_frame_count = 0

    frame_num = int(cap.get(cv2.CAP_PROP_FPS))

    start = time.time()
    while cap.isOpened():
        if frame_count % frame_num == 0:
            year, month, day, hour, minute, second = increment_time(year, month, day, hour, minute, second)

        ret, frame = cap.read()
        if not ret:
            no_frame_count += 1
            date_str = time2str(year, month, day, hour, minute, second)
            logging.warning(f"time: {date_str}, frame: {frame_count}, No frame read!")

            if no_frame_count == 180:
                logging.error("Failed to read frame multiple times, stopping video processing")
                break

            continue

        if frame_count % 3 == 0:
            detect_number = process_frame(frame, model_local, crop_xywh)
            date_str = time2str(year, month, day, hour, minute, second)
            frame_data = {
                'frame': frame_count,
                'number': detect_number,
                'date': date_str
            }
            data.append(frame_data)

        frame_count += 1

    cap.release()
    end = time.time()

    logging.info(f"Video processing completed for file: {filepath}. Total frames: {frame_count}, Processing time: {end - start} seconds")

    df = pd.DataFrame(data)
    measure_df = analyze_number_date(df)
    return measure_df
