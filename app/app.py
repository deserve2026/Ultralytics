import os, sys, time, cv2, json, mimetypes, socket
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, Response, abort
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# 创建必要的文件夹
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局变量
current_model = None
model_info = {
    'loaded': False,
    'model_path': None,
    'model_type': None,
    'classes': []
}
progress_info = {
    'running': False,
    'progress': 0,
    'processed_frames': 0,
    'total_frames': 0,
    'current_fps': 0.0,
    'message': ''
}


class ObjectDetector:
    """目标检测器类"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            dict: 加载结果
        """
        try:
            file_ext = Path(model_path).suffix.lower()
            
            if file_ext in ['.pt', '.pth']:                
                self.model = YOLO(model_path)
                self.model_type = 'yolov8'
                
                # 获取类别名称
                if hasattr(self.model, 'names'):
                    classes = list(self.model.names.values())
                else:
                    classes = []
                
                return {
                    'success': True,
                    'message': '模型加载成功',
                    'model_type': 'YOLOv8',
                    'classes': classes
                }
                
            elif file_ext == '.onnx':
                # ONNX模型（需要额外实现）
                return {
                    'success': False,
                    'message': 'ONNX模型支持开发中'
                }
                
            else:
                return {
                    'success': False,
                    'message': f'不支持的模型格式: {file_ext}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'模型加载失败: {str(e)}'
            }
    
    def detect_image(self, image_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        检测图片
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            dict: 检测结果
        """
        if self.model is None:
            return {
                'success': False,
                'message': '模型未加载'
            }
        
        try:
            start_time = time.time()
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'message': '图片读取失败'
                }
            
            # 执行检测
            if self.model_type == 'yolov8':
                results = self.model(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )[0]
                
                # 绘制结果
                annotated_image = results.plot()
                
                # 提取检测信息
                boxes = results.boxes
                detections = []
                class_counts = {}
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    class_name = self.model.names[cls_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf, 3),
                        'bbox': [float(x) for x in xyxy]
                    })
                    
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 保存结果图片
                output_name = f'result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                output_path = os.path.join(OUTPUT_FOLDER, output_name)
                cv2.imwrite(output_path, annotated_image)
                
                # 计算统计信息
                process_time = int((time.time() - start_time) * 1000)
                avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
                
                return {
                    'success': True,
                    'output_path': f'/outputs/{output_name}',
                    'detections': detections,
                    'stats': {
                        'total_objects': len(detections),
                        'num_classes': len(class_counts),
                        'class_counts': class_counts,
                        'process_time': process_time,
                        'avg_confidence': round(avg_conf * 100, 1)
                    }
                }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'检测失败: {str(e)}'
            }
    
    def detect_video(self, video_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        检测视频
        
        Args:
            video_path: 视频路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            dict: 检测结果
        """
        if self.model is None:
            return {
                'success': False,
                'message': '模型未加载'
            }
        
        try:
            start_time = time.time()
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    'success': False,
                    'message': '视频打开失败'
                }
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = 25
            if width <= 0 or height <= 0:
                cap.release()
                return {
                    'success': False,
                    'message': '视频尺寸无效'
                }
            progress_info.update({
                'running': True,
                'progress': 0,
                'processed_frames': 0,
                'total_frames': total_frames if total_frames > 0 else 0,
                'current_fps': 0.0,
                'message': '正在处理视频'
            })
            
            # 优先使用浏览器兼容编码，失败时降级，保证结果可播放
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer_candidates = [
                (f'result_{timestamp}.mp4', 'avc1'),
                (f'result_{timestamp}.mp4', 'H264'),
                (f'result_{timestamp}.mp4', 'X264'),
                (f'result_{timestamp}.mp4', 'mp4v'),
                (f'result_{timestamp}.avi', 'MJPG'),
            ]
            out = None
            output_name = None
            output_path = None

            for candidate_name, candidate_codec in writer_candidates:
                candidate_path = os.path.join(OUTPUT_FOLDER, candidate_name)
                fourcc = cv2.VideoWriter_fourcc(*candidate_codec)
                writer = cv2.VideoWriter(candidate_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    out = writer
                    output_name = candidate_name
                    output_path = candidate_path
                    break
                writer.release()

            if out is None:
                cap.release()
                return {
                    'success': False,
                    'message': '无法创建输出视频，请检查编码器支持'
                }
            
            # 处理每一帧
            frame_count = 0
            all_detections = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 执行检测
                if self.model_type == 'yolov8':
                    results = self.model(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )[0]
                    
                    # 绘制结果
                    annotated_frame = results.plot()
                    
                    # 记录检测结果
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names[cls_id]
                        
                        all_detections.append({
                            'frame': frame_count,
                            'class': class_name,
                            'confidence': conf
                        })
                    
                    out.write(annotated_frame)
                
                frame_count += 1
                elapsed = max(time.time() - start_time, 1e-6)
                realtime_fps = frame_count / elapsed
                progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                progress_info.update({
                    'progress': min(progress, 99),
                    'processed_frames': frame_count,
                    'current_fps': round(realtime_fps, 2)
                })
                
                # 进度提示（可选）
                if frame_count % 30 == 0:
                    print(f"处理进度: {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            # 计算统计信息
            process_time = int((time.time() - start_time) * 1000)
            class_counts = {}
            for det in all_detections:
                class_counts[det['class']] = class_counts.get(det['class'], 0) + 1
            
            avg_conf = np.mean([d['confidence'] for d in all_detections]) if all_detections else 0
            
            return {
                'success': True,
                'output_path': f'/outputs/{output_name}',
                'stats': {
                    'total_objects': len(all_detections),
                    'num_classes': len(class_counts),
                    'class_counts': class_counts,
                    'total_frames': total_frames,
                    'process_time': process_time,
                    'avg_confidence': round(avg_conf * 100, 1)
                }
            }
            
        except Exception as e:
            progress_info.update({
                'running': False,
                'message': f'视频检测失败: {str(e)}'
            })
            return {
                'success': False,
                'message': f'视频检测失败: {str(e)}'
            }
        finally:
            if progress_info.get('running'):
                progress_info.update({
                    'running': False,
                    'progress': 100,
                    'message': '处理完成'
                })


# 创建检测器实例
detector = ObjectDetector()


# ============= API路由 =============

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """
    加载模型API
    
    接收参数:
        - model: 模型文件
        
    返回:
        - success: 是否成功
        - message: 消息
        - model_info: 模型信息
    """
    if 'model' not in request.files:
        return jsonify({
            'success': False,
            'message': '未找到模型文件'
        })
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': '文件名为空'
        })
    
    # 保存模型文件
    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({
            'success': False,
            'message': '非法文件名'
        })
    model_path = os.path.join(MODEL_FOLDER, safe_name)
    file.save(model_path)
    
    # 加载模型
    result = detector.load_model(model_path)
    
    if result['success']:
        model_info['loaded'] = True
        model_info['model_path'] = model_path
        model_info['model_type'] = result.get('model_type', 'Unknown')
        model_info['classes'] = result.get('classes', [])
        result['model_path'] = model_path
    
    return jsonify(result)


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    检测API
    
    接收参数:
        - file: 图片或视频文件
        - conf_threshold: 置信度阈值
        - iou_threshold: IOU阈值
        - file_type: 文件类型 (image/video)
        
    返回:
        - success: 是否成功
        - output_path: 输出文件路径
        - detections: 检测结果（图片）
        - stats: 统计信息
    """
    if not model_info['loaded']:
        return jsonify({
            'success': False,
            'message': '模型未加载'
        })
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': '未找到文件'
        })
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': '文件名为空'
        })
    
    # 获取参数
    conf_threshold = float(request.form.get('conf_threshold', 0.5))
    iou_threshold = float(request.form.get('iou_threshold', 0.45))
    file_type = request.form.get('file_type', 'image')
    
    # 保存上传文件
    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({
            'success': False,
            'message': '非法文件名'
        })
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(file_path)
    
    # 执行检测
    if file_type == 'image':
        progress_info.update({
            'running': True,
            'progress': 10,
            'processed_frames': 0,
            'total_frames': 0,
            'current_fps': 0.0,
            'message': '正在处理图片'
        })
        result = detector.detect_image(file_path, conf_threshold, iou_threshold)
        progress_info.update({
            'running': False,
            'progress': 100 if result.get('success') else 0,
            'message': '处理完成' if result.get('success') else result.get('message', '处理失败')
        })
    else:
        result = detector.detect_video(file_path, conf_threshold, iou_threshold)
    
    return jsonify(result)


@app.route('/api/download')
def download():
    """
    下载最新结果
    """
    # 获取最新的输出文件
    output_files = sorted(Path(OUTPUT_FOLDER).glob('result_*'))
    if not output_files:
        return jsonify({
            'success': False,
            'message': '没有可下载的文件'
        })
    
    latest_file = output_files[-1]
    return send_file(latest_file, as_attachment=True)


@app.route('/api/model_info')
def get_model_info():
    """
    获取模型信息
    """
    return jsonify(model_info)


@app.route('/api/detect_progress')
def get_detect_progress():
    """
    获取检测进度
    """
    return jsonify(progress_info)


@app.route('/api/clear_outputs', methods=['POST'])
def clear_outputs():
    """
    清空输出文件夹
    """
    try:
        for file in Path(OUTPUT_FOLDER).glob('*'):
            file.unlink()
        return jsonify({
            'success': True,
            'message': '输出文件已清空'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'清空失败: {str(e)}'
        })


# 静态文件服务
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


@app.route('/outputs/<path:filename>')
def output_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    mimetype, _ = mimetypes.guess_type(file_path)
    ext = Path(file_path).suffix.lower()
    if ext == '.mp4':
        mimetype = 'video/mp4'
    elif ext == '.avi':
        mimetype = 'video/x-msvideo'
    elif ext == '.webm':
        mimetype = 'video/webm'
    return send_file(file_path, mimetype=mimetype, conditional=True)


@app.route('/api/preview_video/<path:filename>')
def preview_video(filename):
    """
    视频预览流（MJPEG），用于浏览器无法直接解码输出视频时兜底显示。
    """
    base_dir = Path(OUTPUT_FOLDER).resolve()
    file_path = (base_dir / filename).resolve()

    if base_dir != file_path and base_dir not in file_path.parents:
        abort(404)
    if not file_path.exists() or not file_path.is_file():
        abort(404)

    def generate_frames():
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            cap.release()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        frame_interval = 1.0 / fps

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
                time.sleep(frame_interval)
        finally:
            cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return jsonify({
        'success': False,
        'message': '上传文件过大（超过 2GB 限制）'
    }), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return jsonify({
            'success': False,
            'message': error.description
        }), error.code
    # 避免前端拿到 HTML 错误页导致 fetch/JSON 解析异常
    app.logger.exception(error)
    return jsonify({
        'success': False,
        'message': f'服务端异常: {str(error)}'
    }), 500


if __name__ == '__main__':
    port = 6001
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    host = os.getenv('FLASK_HOST', '0.0.0.0')

    # 兼容 localhost 优先解析到 ::1 的环境：优先尝试 IPv6 双栈监听
    if host == '0.0.0.0':
        try:
            ipv6_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            if hasattr(socket, 'IPPROTO_IPV6') and hasattr(socket, 'IPV6_V6ONLY'):
                ipv6_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            ipv6_socket.bind(('::', 0))
            ipv6_socket.close()
            host = '::'
        except OSError:
            host = '0.0.0.0'

    print("=" * 50)
    print("🚀 目标检测系统启动中...")
    print("=" * 50)
    print(f"📁 上传文件夹: {UPLOAD_FOLDER}")
    print(f"📁 输出文件夹: {OUTPUT_FOLDER}")
    print(f"📁 模型文件夹: {MODEL_FOLDER}")
    print("=" * 50)
    print(f"🌐 本机访问: http://127.0.0.1:{port}")
    print(f"🌐 本机访问: http://localhost:{port}")
    print("=" * 50)
    
    app.run(
        debug=debug_mode,
        host=host,
        port=port,
        threaded=True,
        use_reloader=False
    )
