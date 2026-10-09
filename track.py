import warnings, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定使用第一张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 跟踪参数官方详解链接：https://docs.ultralytics.com/modes/track/#available-trackers:~:text=is%20BoT%2DSORT.-,Tracking,-To%20run%20the

if __name__ == '__main__':
    # 选择训练好的权重路径
    model_path = 'yolo26n.pt'
    model = YOLO(model_path)
    model.track(source='video.mp4', # 视频路径
                imgsz=640,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml",
                project='track',
                name='exp',
                save=True,
                device=os.environ.get("CUDA_VISIBLE_DEVICES", 0), # 训练设备选择，不在这里设置，在头部设置，详细可以看UserGuide.md中的常见问题第4点
                # conf=0.2, # [置信度阈值] 门槛。只有得分 > 0.2 的框才会被保留，低于这个值的认为是杂讯，丢掉
                # iou=0.7, # [NMS阈值] 交并比门槛。用于去除重叠的框。数值越小，去重越严格（框会变少）；数值越大，允许框重叠越多
                # agnostic_nms=True, # [跨类别去重] 如果设为True，不管是不是同一类，只要框重叠了就去掉。比如“人”和“衣服”重叠，可能会被去掉一个
                # classes=[0], # [指定类别] 过滤器。只检测特定的类。比如 [0] 表示只检测“人”，忽略车、狗等其他物体
                # visualize=True, # [特征可视化] 开启后会保存特征图，用来观察模型到底“看”到了图片的哪些特征
                # line_width=2, # [线宽] 画框的线条粗细。如果图片很大，框看着太细，可以把这个数改大
                # show_conf=False, # [隐藏分数] 设为False后，框上面只显示类别名，不显示 0.85 这种概率分数
                # show_labels=False, # [隐藏标签] 设为False后，框上面什么字都不写，只画一个框
                # save_txt=True, # [保存TXT] 不仅保存图，还把检测到的坐标信息保存成 .txt 文件（用于后续数据分析）
                # save_crop=True, # [保存抠图] 把检测到的目标单独“抠”出来，保存成小图
                # save_frames=True, # [保存帧] 如果输入是视频，把视频的每一帧都作为图片保存下来
                # end2end=False # 如果训练的是NMSFree类型的模型，不想用一对一的头可以设置False
                )