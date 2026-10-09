import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第一张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from prettytable import PrettyTable

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import model_info

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, BOLD, RESET = (
    "\033[91m",
    "\033[92m",
    "\033[94m",
    "\033[93m",
    "\033[38;5;208m",
    "\033[96m",
    "\033[95m",
    "\033[1m",
    "\033[0m",
)


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


def fmt_metric(v):
    return "N/A" if v is None else f"{float(v):.4f}"


def mean_per_class_metric(per_class_branch, key):
    vals = [m.get(key) for m in per_class_branch.values() if m.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _color_text(text, color=GREEN, bold=False):
    style = BOLD if bold else ""
    return f"{style}{color}{text}{RESET}"


def print_highlight_table(table, header_color_value_cols=None, color_first_col=True):
    """高亮打印 PrettyTable，终端可读性更好。."""
    header_color_value_cols = set(header_color_value_cols or [])
    highlighted = PrettyTable()
    highlighted.title = _color_text(table.title, CYAN, bold=True) if table.title else table.title
    highlighted.field_names = [_color_text(name, YELLOW, bold=True) for name in table.field_names]

    for row in table._rows:
        row = list(row)
        is_avg = bool(row) and isinstance(row[0], str) and "all(" in row[0]
        colored_row = []
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if is_avg:
                colored_row.append(_color_text(cell_str, ORANGE, bold=True))
            elif i == 0 and color_first_col:
                colored_row.append(_color_text(cell_str, BLUE, bold=False))
            elif i in header_color_value_cols:
                colored_row.append(_color_text(cell_str, YELLOW, bold=False))
            else:
                colored_row.append(_color_text(cell_str, GREEN, bold=False))
        highlighted.add_row(colored_row)

    print(highlighted)


if __name__ == "__main__":
    # 选择训练好的权重路径
    model_path = "yolo26n.pt"
    # 设置用于计算指标的图像尺寸
    imgsz = 640

    model = YOLO(model_path)
    result = model.val(
        data="/root/dataset/dataset_visdrone/data.yaml",
        split="test",  # split可以选择train、val、test 根据自己的数据集情况来选择.
        imgsz=imgsz,
        batch=16,
        rect=False,  # 验证时统一固定imgsz x imgsz 做 letterbox，避免一些改进在验证的时候会报尺寸问题
        auto_coco_eval=True,  # 一步到位计算COCO指标
        # iou=0.7,
        # save_json=True,
        project="val",
        name="exp",
        device=os.environ.get(
            "CUDA_VISIBLE_DEVICES", 0
        ),  # 训练设备选择，不在这里设置，在头部设置，详细可以看UserGuide.md中的常见问题第4点
        # end2end=False # 如果训练的是NMSFree类型的模型，不想用一对一的头可以设置False
    )

    length = result.box.p.size
    model_names = list(result.names.values())
    preprocess_time_per_image = result.speed["preprocess"]
    inference_time_per_image = result.speed["inference"]
    postprocess_time_per_image = result.speed["postprocess"]
    all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

    n_l, n_p, n_g, flops = model_info(model.model, imgsz=imgsz)

    model_info_table = PrettyTable()
    model_info_table.title = "Model Info"
    model_info_table.field_names = [
        "GFLOPs",
        "Parameters",
        "前处理时间/一张图",
        "推理时间/一张图",
        "后处理时间/一张图",
        "FPS(前处理+模型推理+后处理)",
        "FPS(推理)",
        "Model File Size",
    ]
    model_info_table.add_row(
        [
            f"{flops:.1f}",
            f"{n_p:,}",
            f"{preprocess_time_per_image / 1000:.6f}s",
            f"{inference_time_per_image / 1000:.6f}s",
            f"{postprocess_time_per_image / 1000:.6f}s",
            f"{1000 / all_time_per_image:.2f}",
            f"{1000 / inference_time_per_image:.2f}",
            f"{get_weight_size(model_path)}MB",
        ]
    )

    for _ in range(5):
        LOGGER.info(f"{BOLD}{ORANGE}{'-' * 20}论文上的数据以以下结果为准{'-' * 20}{RESET}")

    print_highlight_table(model_info_table, color_first_col=False)

    yolo_metrics_table = PrettyTable()
    yolo_metrics_table.title = "YOLO Metrics"
    if model.task == "detect" or model.task == "obb":
        yolo_metrics_table.field_names = [
            "Class Name",
            "Box (Precision",
            "Recall",
            "F1-Score",
            "mAP50",
            "mAP75",
            "mAP50-95)",
        ]
        for idx in range(length):
            yolo_metrics_table.add_row(
                [
                    model_names[idx],
                    f"{result.box.p[idx]:.4f}",
                    f"{result.box.r[idx]:.4f}",
                    f"{result.box.f1[idx]:.4f}",
                    f"{result.box.ap50[idx]:.4f}",
                    f"{result.box.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                    f"{result.box.ap[idx]:.4f}",
                ]
            )
        yolo_metrics_table.add_row(
            [
                "all(平均数据)",
                f"{result.results_dict['metrics/precision(B)']:.4f}",
                f"{result.results_dict['metrics/recall(B)']:.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
            ]
        )
    elif model.task == "segment":
        yolo_metrics_table.field_names = [
            "Class Name",
            "Precision(Box)",
            "Recall(Box)",
            "F1-Score(Box)",
            "mAP50(Box)",
            "mAP75(Box)",
            "mAP50-95(Box)",
            "Precision(Seg)",
            "Recall(Seg)",
            "F1-Score(Seg)",
            "mAP50(Seg)",
            "mAP75(Seg)",
            "mAP50-95(Seg)",
        ]
        for idx in range(length):
            yolo_metrics_table.add_row(
                [
                    model_names[idx],
                    f"{result.box.p[idx]:.4f}",
                    f"{result.box.r[idx]:.4f}",
                    f"{result.box.f1[idx]:.4f}",
                    f"{result.box.ap50[idx]:.4f}",
                    f"{result.box.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                    f"{result.box.ap[idx]:.4f}",
                    f"{result.seg.p[idx]:.4f}",
                    f"{result.seg.r[idx]:.4f}",
                    f"{result.seg.f1[idx]:.4f}",
                    f"{result.seg.ap50[idx]:.4f}",
                    f"{result.seg.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                    f"{result.seg.ap[idx]:.4f}",
                ]
            )
        yolo_metrics_table.add_row(
            [
                "all(平均数据)",
                f"{result.results_dict['metrics/precision(B)']:.4f}",
                f"{result.results_dict['metrics/recall(B)']:.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
                f"{result.results_dict['metrics/precision(M)']:.4f}",
                f"{result.results_dict['metrics/recall(M)']:.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict['metrics/mAP50(M)']:.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.results_dict['metrics/mAP50-95(M)']:.4f}",
            ]
        )
    elif model.task == "pose":
        yolo_metrics_table.field_names = [
            "Class Name",
            "Precision(Box)",
            "Recall(Box)",
            "F1-Score(Box)",
            "mAP50(Box)",
            "mAP75(Box)",
            "mAP50-95(Box)",
            "Precision(Pose)",
            "Recall(Pose)",
            "F1-Score(Pose)",
            "mAP50(Pose)",
            "mAP75(Pose)",
            "mAP50-95(Pose)",
        ]
        for idx in range(length):
            yolo_metrics_table.add_row(
                [
                    model_names[idx],
                    f"{result.box.p[idx]:.4f}",
                    f"{result.box.r[idx]:.4f}",
                    f"{result.box.f1[idx]:.4f}",
                    f"{result.box.ap50[idx]:.4f}",
                    f"{result.box.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                    f"{result.box.ap[idx]:.4f}",
                    f"{result.pose.p[idx]:.4f}",
                    f"{result.pose.r[idx]:.4f}",
                    f"{result.pose.f1[idx]:.4f}",
                    f"{result.pose.ap50[idx]:.4f}",
                    f"{result.pose.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                    f"{result.pose.ap[idx]:.4f}",
                ]
            )
        yolo_metrics_table.add_row(
            [
                "all(平均数据)",
                f"{result.results_dict['metrics/precision(B)']:.4f}",
                f"{result.results_dict['metrics/recall(B)']:.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
                f"{result.results_dict['metrics/precision(P)']:.4f}",
                f"{result.results_dict['metrics/recall(P)']:.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict['metrics/mAP50(P)']:.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.results_dict['metrics/mAP50-95(P)']:.4f}",
            ]
        )

    print_highlight_table(yolo_metrics_table)

    coco_metrics_table = None
    if model.task != "obb":
        coco_results = getattr(result, "coco_results_dict", {}) or {}
        per_class = coco_results.get("per_class", {})
        if per_class:
            coco_metrics_table = PrettyTable()
            coco_metrics_table.title = "COCO Metrics"
            if model.task == "detect":
                coco_metrics_table.field_names = [
                    "Class Name",
                    "AP50(Box)",
                    "AP75(Box)",
                    "AP50-95(Box)",
                    "APs(Box)",
                    "APm(Box)",
                    "APl(Box)",
                ]
                box_pc = per_class.get("B", {})
                for name in model_names:
                    m = box_pc.get(name, {})
                    coco_metrics_table.add_row(
                        [
                            name,
                            fmt_metric(m.get("AP50")),
                            fmt_metric(m.get("AP75")),
                            fmt_metric(m.get("AP50-95")),
                            fmt_metric(m.get("APs")),
                            fmt_metric(m.get("APm")),
                            fmt_metric(m.get("APl")),
                        ]
                    )
                coco_metrics_table.add_row(
                    [
                        "all(平均数据)",
                        fmt_metric(mean_per_class_metric(box_pc, "AP50")),
                        fmt_metric(mean_per_class_metric(box_pc, "AP75")),
                        fmt_metric(mean_per_class_metric(box_pc, "AP50-95")),
                        fmt_metric(mean_per_class_metric(box_pc, "APs")),
                        fmt_metric(mean_per_class_metric(box_pc, "APm")),
                        fmt_metric(mean_per_class_metric(box_pc, "APl")),
                    ]
                )
            elif model.task == "segment":
                coco_metrics_table.field_names = [
                    "Class Name",
                    "AP50(Box)",
                    "AP75(Box)",
                    "AP50-95(Box)",
                    "APs(Box)",
                    "APm(Box)",
                    "APl(Box)",
                    "AP50(Seg)",
                    "AP75(Seg)",
                    "AP50-95(Seg)",
                    "APs(Seg)",
                    "APm(Seg)",
                    "APl(Seg)",
                ]
                box_pc = per_class.get("B", {})
                seg_pc = per_class.get("M", {})
                for name in model_names:
                    mb = box_pc.get(name, {})
                    ms = seg_pc.get(name, {})
                    coco_metrics_table.add_row(
                        [
                            name,
                            fmt_metric(mb.get("AP50")),
                            fmt_metric(mb.get("AP75")),
                            fmt_metric(mb.get("AP50-95")),
                            fmt_metric(mb.get("APs")),
                            fmt_metric(mb.get("APm")),
                            fmt_metric(mb.get("APl")),
                            fmt_metric(ms.get("AP50")),
                            fmt_metric(ms.get("AP75")),
                            fmt_metric(ms.get("AP50-95")),
                            fmt_metric(ms.get("APs")),
                            fmt_metric(ms.get("APm")),
                            fmt_metric(ms.get("APl")),
                        ]
                    )
                coco_metrics_table.add_row(
                    [
                        "all(平均数据)",
                        fmt_metric(mean_per_class_metric(box_pc, "AP50")),
                        fmt_metric(mean_per_class_metric(box_pc, "AP75")),
                        fmt_metric(mean_per_class_metric(box_pc, "AP50-95")),
                        fmt_metric(mean_per_class_metric(box_pc, "APs")),
                        fmt_metric(mean_per_class_metric(box_pc, "APm")),
                        fmt_metric(mean_per_class_metric(box_pc, "APl")),
                        fmt_metric(mean_per_class_metric(seg_pc, "AP50")),
                        fmt_metric(mean_per_class_metric(seg_pc, "AP75")),
                        fmt_metric(mean_per_class_metric(seg_pc, "AP50-95")),
                        fmt_metric(mean_per_class_metric(seg_pc, "APs")),
                        fmt_metric(mean_per_class_metric(seg_pc, "APm")),
                        fmt_metric(mean_per_class_metric(seg_pc, "APl")),
                    ]
                )
            elif model.task == "pose":
                coco_metrics_table.field_names = [
                    "Class Name",
                    "AP50(Box)",
                    "AP75(Box)",
                    "AP50-95(Box)",
                    "APs(Box)",
                    "APm(Box)",
                    "APl(Box)",
                    "AP50(Pose)",
                    "AP75(Pose)",
                    "AP50-95(Pose)",
                    "APs(Pose)",
                    "APm(Pose)",
                    "APl(Pose)",
                ]
                box_pc = per_class.get("B", {})
                pose_pc = per_class.get("P", {})
                for name in model_names:
                    mb = box_pc.get(name, {})
                    mp = pose_pc.get(name, {})
                    coco_metrics_table.add_row(
                        [
                            name,
                            fmt_metric(mb.get("AP50")),
                            fmt_metric(mb.get("AP75")),
                            fmt_metric(mb.get("AP50-95")),
                            fmt_metric(mb.get("APs")),
                            fmt_metric(mb.get("APm")),
                            fmt_metric(mb.get("APl")),
                            fmt_metric(mp.get("AP50")),
                            fmt_metric(mp.get("AP75")),
                            fmt_metric(mp.get("AP50-95")),
                            fmt_metric(mp.get("APs")),
                            fmt_metric(mp.get("APm")),
                            fmt_metric(mp.get("APl")),
                        ]
                    )
                coco_metrics_table.add_row(
                    [
                        "all(平均数据)",
                        fmt_metric(mean_per_class_metric(box_pc, "AP50")),
                        fmt_metric(mean_per_class_metric(box_pc, "AP75")),
                        fmt_metric(mean_per_class_metric(box_pc, "AP50-95")),
                        fmt_metric(mean_per_class_metric(box_pc, "APs")),
                        fmt_metric(mean_per_class_metric(box_pc, "APm")),
                        fmt_metric(mean_per_class_metric(box_pc, "APl")),
                        fmt_metric(mean_per_class_metric(pose_pc, "AP50")),
                        fmt_metric(mean_per_class_metric(pose_pc, "AP75")),
                        fmt_metric(mean_per_class_metric(pose_pc, "AP50-95")),
                        fmt_metric(mean_per_class_metric(pose_pc, "APs")),
                        fmt_metric(mean_per_class_metric(pose_pc, "APm")),
                        fmt_metric(mean_per_class_metric(pose_pc, "APl")),
                    ]
                )
            if coco_metrics_table is not None:
                print_highlight_table(coco_metrics_table)
        else:
            LOGGER.warning("当前任务未返回 COCO per-class 指标，跳过 COCO 表格输出。")

    with open(result.save_dir / "paper_data.txt", "w+", errors="ignore", encoding="utf-8") as f:
        f.write(str(model_info_table))
        f.write("\n")
        f.write(str(yolo_metrics_table))
        if coco_metrics_table is not None:
            f.write("\n")
            f.write(str(coco_metrics_table))

    for _ in range(5):
        LOGGER.info(f"{BOLD}{ORANGE}{'-' * 20}结果已保存至 {result.save_dir}/paper_data.txt...{'-' * 20}{RESET}")
