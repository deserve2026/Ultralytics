import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第一张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' # 指定使用第三、四张显卡进行多卡训练
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    yaml_path = "ultralytics/cfg/models/26/yolo26.yaml"

    # 初始化 YOLO 模型，从 yaml 配置文件构建网络结构
    model = YOLO(yaml_path)
    # model.load('yolo26.pt') # 加载预训练权重，一般都不建议加载
    model.train(
        data="D:/experimental_data/mend26s/dataset_0918_yolo/data.yaml",  # 新数据集: dataset_0918 (687张, 去重+连续块划分, 两类 fish/suspect)
        cache=False,  # 是否缓存图像到内存以加快训练速度。False=不缓存，True=缓存到RAM(很吃内存，内存少的慎开)，'disk'=缓存到磁盘(吃硬盘空间)
        imgsz=640,  # 输入图像尺寸（像素）
        epochs=100,  # 训练总轮数
        batch=8,  # 批次大小
        close_mosaic=0,  # 最后多少个 epoch 关闭 Mosaic 数据增强。设置 0 代表全程开启 Mosaic 训练
        workers=4,  # 数据加载的工作线程数。Windows 下出现卡顿或奇怪错误可尝试设置为 0
        device=os.environ.get(
            "CUDA_VISIBLE_DEVICES", 0
        ),  # 训练设备选择，不在这里设置，在头部设置，详细可以看UserGuide.md中的常见问题第4点
        optimizer="MuSGD"
        if "yolo26" in yaml_path
        else "SGD",  # 优化器选择。YOLO26 使用官方推荐的 MuSGD，其他模型使用 SGD
        patience=50,  # 早停机制的耐心值。连续 50 个 epoch 验证指标未提升则停止训练。设置 0 关闭早停
        # resume=True, # 断点续训，需要在 YOLO 初始化时加载 last.pt 权重文件
        amp=True,  # 是否启用自动混合精度（Automatic Mixed Precision）训练，默认为 True | loss出现nan可以关闭amp
        # fraction=0.2, # 设置0.2代表只选择百分之20的数据进行训练
        cos_lr=False,  # 是否使用余弦退火学习率调度器，默认为 False
        save_period=-1,  # 每隔多少个 epoch 保存一次 checkpoint（默认 -1 表示禁用，仅保存最好和最后的）
        project="D:/experimental_result/mend26/yazhou/data_0916",  # 训练结果保存的项目目录
        name="exp_0918",  # 本次实验的名称，（若已存在则自动创建 exp2, exp3...）
        plots=True,  # 保存训练曲线、混淆矩阵、标签分布等图
        show_labels=False,  # 结果示意图(batch/val 预览图)里的目标框【不显示类别标签】
        # trainer=AFSSDetectionTrainer,
        # afss=True, # 开启 AFSS
        # afss_save_refresh_json=False,
        # afss_warmup_epochs=20, # 前 20 个 epoch 使用全量训练集 warmup
        # afss_update_interval=5, # 每隔 5 个 epoch 刷新一次图像难度状态
        # afss_easy_ratio=0.02, # easy 样本每轮保留 2%
        # afss_moderate_ratio=0.40, # moderate 样本每轮保留 40%
        # afss_easy_forced_gap=10, # easy 样本超过 10 个 epoch 未使用则强制回看
        # afss_moderate_forced_gap=3, # moderate 样本超过 3 个 epoch 未使用则强制覆盖
        # afss_thresholds={
        #     "detect": [0.55, 0.85],
        #     "obb": [0.55, 0.85],
        #     "segment": [0.55, 0.85],
        #     "pose": [0.55, 0.85],
        # },
        # -------------------- LOSS部分(更多解释可以看LOSS-UserGuide.md) --------------------
        cls_loss="bce",  # 分类损失类型可选：bce, slide, ema_slide, focal, varifocal, qualityfocal
        iou_loss="ciou",  # IoU损失可选：基础 iou/giou/diou/ciou/eiou/siou/shapeiou/piou/piou2；组合 inner_<base>/focaler_<base>；MPD mpdiou/inner_mpdiou/focaler_mpdiou；Wise wiseiou[_inner|_focaler]_<variant>
        iou_aux="none",  # IoU辅助分支可选：none, gcd, nwd（none 表示关闭辅助分支）
        iou_aux_ratio=0.5,  # IoU主损失与辅助分支混合系数（0~1），仅在 iou_aux != none 时生效
    )
