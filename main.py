import cv2
import time
import mediapipe as mp
import os
import urllib.request
import sys
import threading
import queue

# 1. 硬编码 33 个关键点的连接关系
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

# ================= 辅助模块：模型下载管理 =================
MODELS_INFO = {
    "lite": ("pose_landmarker_lite.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "full": ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    "heavy": ("pose_landmarker_heavy.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
}

def dl_progress_hook(count, block_size, total_size):
    """显示下载进度条"""
    percent = int(count * block_size * 100 / total_size)
    if percent <= 100:
        sys.stdout.write(f"\r下载进度: {percent}%")
        sys.stdout.flush()

def check_models():
    """检查本地模型，返回可用和缺失的模型列表"""
    available, missing = [], []
    for key, (filename, _) in MODELS_INFO.items():
        if os.path.exists(filename):
            available.append(key)
        else:
            missing.append(key)
    return available, missing

# ================= 辅助模块：多线程视频流提取 =================
class VideoStream:
    """包装了原生 VideoCapture，支持开启多线程队列读取以提升帧率"""
    def __init__(self, src, is_camera=False, use_threading=False):
        self.cap = cv2.VideoCapture(src)
        self.is_opened = self.cap.isOpened()
        self.use_threading = use_threading
        self.is_camera = is_camera
        
        if not self.is_opened:
            return

        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[摄像头] 已自动协商至最高分辨率: {actual_w} x {actual_h}")

        if self.use_threading:
            self.q = queue.Queue(maxsize=15) 
            self.stopped = False
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        """后台线程：不断读取视频帧放入队列"""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return
            
            if self.is_camera:
                # 实时摄像头模式下的“零延迟”策略：清空积压，只保留最新帧
                while not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break
                self.q.put((ret, frame))
            else:
                # 本地视频模式下的“不丢帧”策略
                if not self.q.full():
                    self.q.put((ret, frame))
                else:
                    time.sleep(0.005)

    def read(self):
        """读取一帧画面"""
        if self.use_threading:
            if self.stopped and self.q.empty():
                return False, None
            while self.q.empty():
                if self.stopped:
                    return False, None
                time.sleep(0.001)
            return self.q.get()
        else:
            return self.cap.read()

    def release(self):
        if self.use_threading:
            self.stopped = True
            if hasattr(self, 'thread'):
                self.thread.join()
        else:
            self.cap.release()

    def get(self, prop_id):
        return self.cap.get(prop_id)

# ================= 核心处理模块 =================
def show_fit_window(window_name, image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        display_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        display_image = image
    cv2.imshow(window_name, display_image)

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if not detection_result.pose_landmarks:
        return annotated_image

    h, w, _ = annotated_image.shape
    for pose_landmarks in detection_result.pose_landmarks:
        keypoints = []
        for landmark in pose_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            keypoints.append((cx, cy))
            cv2.circle(annotated_image, (cx, cy), 4, (24, 200, 150), -1)

        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                pt1 = keypoints[start_idx]
                pt2 = keypoints[end_idx]
                cv2.line(annotated_image, pt1, pt2, (250, 100, 10), 2)
                
    return annotated_image

def process_image(model_path, image_path, save_output=False, model_name=""):
    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 {image_path}")
        return

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        cv_mat = cv2.imread(image_path)
        rgb_frame = cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = landmarker.detect(mp_image)
        result_img = draw_landmarks_on_image(rgb_frame, detection_result)
        
        if save_output:
            base, ext = os.path.splitext(image_path)
            suffix = f"_{model_name}" if model_name else ""
            out_path = f"{base}{suffix}_output{ext}"
            cv2.imwrite(out_path, result_img)
            print(f"已保存至: {out_path}")

        show_fit_window('MediaPipe Pose Estimation', result_img)
        print("处理完成。按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video_stream(model_path, source, is_camera=False, save_output=False, use_threading=False, headless=False, model_name=""):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1
    )

    stream = VideoStream(source, is_camera=is_camera, use_threading=use_threading)
    if not stream.is_opened:
        print(f"错误：无法打开视频源 {source}")
        return

    out_writer = None
    if save_output:
        suffix = f"_{model_name}" if model_name else ""
        if is_camera:
            out_path = f"camera{suffix}_output_{int(time.time())}.mp4"
            fps = 30.0 
        else:
            base, ext = os.path.splitext(str(source))
            out_path = f"{base}{suffix}_output.mp4"
            fps = stream.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps:
                fps = 30.0
        print(f"准备保存视频至: {out_path}")

    start_time = time.monotonic()
    prev_frame_time = start_time
    last_timestamp_ms = -1
    frame_count = 0

    print("\n" + "="*30)
    if headless:
        print("🚀 已启动【无头模式 (极限帧率)】，将不会显示画面。")
        print("💡 提示: 随时按 Ctrl + C 结束并保存！")
    else:
        print("🚀 正在运行，按 'q' 键退出。")
    print("="*30 + "\n")

    with PoseLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                success, frame = stream.read()
                if not success:
                    if not is_camera:
                        print("\n视频流处理结束。")
                    break

                if is_camera:
                    frame = cv2.flip(frame, 1)

                if save_output and out_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                result_frame = draw_landmarks_on_image(rgb_frame, detection_result)

                if save_output and out_writer is not None:
                    out_writer.write(result_frame)

                new_frame_time = time.monotonic()
                current_fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
                prev_frame_time = new_frame_time
                frame_count += 1

                # === 显示与刷新逻辑 ===
                if headless:
                    if frame_count % 30 == 0:
                        print(f"正在后台极速处理... 当前帧率: {int(current_fps)} FPS")
                else:
                    cv2.putText(result_frame, f'FPS: {int(current_fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    show_fit_window('MediaPipe Pose Estimation', result_frame)
                    delay = 1 
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\n[中断] 接收到 Ctrl+C，正在终止程序...")

    # --- 新增：核心统计数据输出 ---
    total_time = time.monotonic() - start_time
    if frame_count > 0 and total_time > 0:
        avg_fps = frame_count / total_time
        print("\n" + "="*35)
        print("📊 处理统计报告")
        print("="*35)
        print(f"使用模型: {model_name.capitalize()}")
        print(f"总处理帧数: {frame_count} 帧")
        print(f"总耗时:   {total_time:.2f} 秒")
        print(f"平均帧率:   {avg_fps:.2f} FPS")
        print("="*35 + "\n")

    stream.release()
    if out_writer is not None:
        out_writer.release()
        print(f"✅ 视频已成功保存至: {out_path}")
    cv2.destroyAllWindows()

def main():
    while True:
        available_models, missing_models = check_models()
        
        print("\n" + "="*45)
        print("🤖 MediaPipe 姿势检测综合工具 (高性能版)")
        print("="*45)
        
        if missing_models:
            print("0. ⬇️  下载缺失的模型 (Lite/Full/Heavy)")
        print("1. 🖼️  测试单张图片")
        print("2. 🎬  测试本地视频文件")
        print("3. 📷  使用 USB 摄像头实时捕捉")
        print("q. ❌  退出程序")
        print("="*45)
        
        choice = input("请输入选项: ").strip().lower()
        
        if choice == 'q':
            print("再见！")
            break
            
        # =============== 下载模型逻辑 ===============
        if choice == '0' and missing_models:
            print("\n发现以下缺失模型：")
            for i, m in enumerate(missing_models):
                print(f"{i+1}. {m.capitalize()} 模型")
            print("a. 全部下载")
            
            dl_choice = input("\n请选择要下载的项 (例如输入 1 或 a): ").strip().lower()
            to_download = []
            
            if dl_choice == 'a':
                to_download = missing_models
            elif dl_choice.isdigit() and 1 <= int(dl_choice) <= len(missing_models):
                to_download.append(missing_models[int(dl_choice)-1])
            else:
                print("无效输入！")
                continue
                
            for m in to_download:
                filename, url = MODELS_INFO[m]
                print(f"\n开始下载 {filename} ...")
                try:
                    urllib.request.urlretrieve(url, filename, dl_progress_hook)
                    print(f"\n✅ {filename} 下载成功！")
                except Exception as e:
                    print(f"\n❌ 下载失败: {e}")
            continue

        # =============== 运行检测逻辑 ===============
        if choice in ['1', '2', '3']:
            if not available_models:
                print("\n⚠️ 本地未找到任何模型文件！请先输入 0 进行下载。")
                continue
                
            # 选择要使用的模型
            print("\n请选择要加载的模型精度：")
            for i, m in enumerate(available_models):
                print(f"{i+1}. {m.capitalize()}")
            m_choice = input("请输入序号: ").strip()
            
            if not (m_choice.isdigit() and 1 <= int(m_choice) <= len(available_models)):
                print("无效选择，返回主菜单。")
                continue
                
            selected_model = available_models[int(m_choice)-1] 
            model_path = MODELS_INFO[selected_model][0]
            
            # 单张图片处理
            if choice == '1':
                img_path = input("\n请输入图片路径: ").strip().strip('"').strip("'")
                save_choice = input("是否保存照片？(y/n) [默认 n]: ").strip().lower() == 'y'
                process_image(model_path, img_path, save_output=save_choice, model_name=selected_model)
                
            # 视频与摄像头处理
            elif choice in ['2', '3']:
                if choice == '3':
                    src_input = input("\n请输入摄像头设备号 (直接回车默认为 0): ").strip()
                    src = int(src_input) if src_input.isdigit() else 0
                    is_camera = True
                else:
                    src = input("\n请输入视频路径: ").strip().strip('"').strip("'")
                    is_camera = False
                    
                save_choice = input("是否保存视频结果？(y/n) [默认 n]: ").strip().lower() == 'y'
                
                print("\n请选择性能模式：")
                print("1. 标准模式 (单线程串行 + UI显示)")
                print("2. 无头模式 (单线程串行 + 隐藏UI极限运算)")
                print("3. 多线程模式 (独立线程读取缓存 + UI显示)")
                print("4. 性能怪兽模式 (独立线程读取缓存 + 隐藏UI极限运算)")
                mode_choice = input("请输入选项 (1/2/3/4) [默认 1]: ").strip()
                
                headless = mode_choice in ['2', '4']
                use_threading = mode_choice in ['3', '4']
                
                process_video_stream(model_path, src, is_camera, save_choice, use_threading, headless, model_name=selected_model)

if __name__ == '__main__':
    main()