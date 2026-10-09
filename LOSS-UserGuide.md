# LOSS User Guide

本文档说明当前项目中自定义 loss 开关的统一用法，面向训练时通过 `yolo ... arg=value` 或 `model.train(**kwargs)` 直接配置的用户。

## 1. 设计目标

- 不改代码即可切换分类损失与 IoU 损失。
- 默认配置保持兼容（不显式设置时行为与原版一致）。
- 提供可组合命名：`inner_*`、`focaler_*`、`wiseiou_*`。

## 2. 默认行为（兼容）

- `cls_loss=bce`
- `iou_loss=ciou`
- `iou_aux=none`

只要你不改这些参数，训练行为保持原有基线。

## 3. 分类损失（`cls_loss`）

可选值：

- `bce`
- `slide`
- `ema_slide`
- `focal`
- `varifocal`
- `qualityfocal`

相关参数：

- `slide_auto_iou_min`、`slide_delta`
- `ema_decay`、`ema_tau`
- `focal_gamma`、`focal_alpha`
- `varifocal_alpha`、`varifocal_gamma`
- `qfl_beta`

## 4. IoU 损失（`iou_loss`）命名规则

### 4.1 基础形式

- `iou`
- `giou`
- `diou`
- `ciou`
- `eiou`
- `siou`
- `shapeiou`
- `piou`
- `piou2`

### 4.2 Inner 形式

格式：`inner_<base>`

例如：

- `inner_iou`
- `inner_giou`
- `inner_diou`
- `inner_ciou`
- `inner_eiou`
- `inner_siou`
- `inner_shapeiou`
- `inner_piou`
- `inner_piou2`

### 4.3 Focaler 形式

格式：`focaler_<base>`

例如：

- `focaler_iou`
- `focaler_giou`
- `focaler_diou`
- `focaler_ciou`
- `focaler_eiou`
- `focaler_siou`
- `focaler_shapeiou`
- `focaler_piou`
- `focaler_piou2`

### 4.4 MPDIoU 家族

- `mpdiou`
- `inner_mpdiou`
- `focaler_mpdiou`

### 4.5 WiseIoU 家族

格式：

- `wiseiou`（等价 `wiseiou_wiou`）
- `wiseiou_<variant>`
- `wiseiou_inner_<variant>`
- `wiseiou_focaler_<variant>`

`<variant>` 可选：

- `iou`
- `wiou`
- `giou`
- `diou`
- `ciou`
- `eiou`
- `siou`
- `shapeiou`
- `piou`
- `piou2`
- `mpdiou`

例如：

- `wiseiou_diou`
- `wiseiou_inner_diou`
- `wiseiou_focaler_diou`
- `wiseiou_shapeiou`

## 5. IoU 辅助混合（`iou_aux`）

可选值：

- `none`
- `gcd`
- `nwd`

混合系数：

- `iou_aux_ratio`，总损失为：`iou_aux_ratio * iou_loss + (1 - iou_aux_ratio) * aux_loss`

## 6. 其他 IoU 相关参数

- `inner_iou_ratio`：`inner_*` 系列比例
- `focaler_d`、`focaler_u`：`focaler_*` 映射区间
- `shapeiou_scale`：`shapeiou` 参数
- `piou_lambda`：`piou2` 参数
- `wiseiou_monotonous`：WiseIoU 单调模式开关

## 7. CLI 示例

### 7.1 保持默认（兼容基线）

```bash
yolo detect train data=coco8.yaml model=yolo26n.pt
```

### 7.2 Inner-DIoU + GCD 混合

```bash
yolo detect train data=coco8.yaml model=yolo26n.pt \
  cls_loss=focal focal_gamma=1.5 focal_alpha=0.25 \
  iou_loss=inner_diou iou_aux=gcd iou_aux_ratio=0.5 \
  inner_iou_ratio=0.7
```

### 7.3 WiseIoU Inner-DIoU + Quality Focal

```bash
yolo detect train data=coco8.yaml model=yolo26n.pt \
  cls_loss=qualityfocal qfl_beta=2.0 \
  iou_loss=wiseiou_inner_diou wiseiou_monotonous=False \
  iou_aux=none
```

## 8. Python 示例

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data="coco8.yaml",
    cls_loss="varifocal",
    varifocal_alpha=0.75,
    varifocal_gamma=2.0,
    iou_loss="focaler_siou",
    focaler_d=0.0,
    focaler_u=0.95,
    iou_aux="nwd",
    iou_aux_ratio=0.5,
)
```

## 9. 训练日志中的 LossConfig 说明

### 9.1 输出时机

- 在检测任务训练中，`v8DetectionLoss` 初始化完成后会输出一条 `LossConfig` 日志。
- 该日志用于确认当前实际生效的 `cls_loss` / `iou_loss` / `iou_aux` 组合以及关键参数。
- 仅在主进程输出（`RANK in {-1, 0}`），避免多卡重复刷屏。

### 9.2 输出格式

日志是“固定字段 + 条件字段”的单行摘要，格式类似：

```text
LossConfig cls_loss=..., iou_loss=..., iou_aux=...; iou_family=..., iou_variant=..., ...
```

说明：

- 实际终端中 `LossConfig` 会通过 `LOGGER + colorstr` 高亮显示。
- 字段名与配置键保持一致，可直接对照 `yolo ... key=value` 或 `model.train(**kwargs)` 参数。

### 9.3 字段释义

固定字段（每次都会出现）：

- `cls_loss`：当前分类损失类型。
- `iou_loss`：当前 IoU 主损失命名。
- `iou_aux`：IoU 辅助损失类型（`none/gcd/nwd`）。
- `iou_family`：解析后的 IoU 家族（`base/inner/focaler/mpd/wise`）。
- `iou_variant`：解析后的具体变体（如 `ciou/diou/mpdiou`）。

条件字段（仅在对应组合下出现）：

- `slide_auto_iou_min`、`slide_delta`：`slide` / `ema_slide`。
- `ema_decay`、`ema_tau`：`ema_slide`。
- `focal_gamma`、`focal_alpha`：`focal`。
- `varifocal_alpha`、`varifocal_gamma`：`varifocal`。
- `qfl_beta`：`qualityfocal`。
- `iou_aux_ratio`：`iou_aux != none`。
- `inner_iou_ratio`：inner 系列（含 wise inner、inner_mpdiou）。
- `focaler_d`、`focaler_u`：focaler 系列（含 wise focaler、focaler_mpdiou）。
- `shapeiou_scale`：`shapeiou` 变体。
- `piou_lambda`：`piou` / `piou2` 变体。
- `wiseiou_monotonous`：wise 系列。

### 9.4 最小覆盖样例（推荐）

以下 6 条示例可用最小数量覆盖大多数 loss 分支。你可以直接对照自己的训练日志，快速确认配置是否按预期生效。

1) 基线默认（兼容原版）：

```text
LossConfig cls_loss=bce, iou_loss=ciou, iou_aux=none; iou_family=base, iou_variant=ciou
```

2) Slide + Inner + GCD（覆盖 slide、inner、iou_aux_ratio）：

```text
LossConfig cls_loss=slide, iou_loss=inner_diou, iou_aux=gcd; iou_family=inner, iou_variant=diou, slide_auto_iou_min=0.200, slide_delta=0.100, iou_aux_ratio=0.500, inner_iou_ratio=0.700
```

3) EMA Slide + Focaler + NWD（覆盖 ema_*、focaler_*、iou_aux_ratio）：

```text
LossConfig cls_loss=ema_slide, iou_loss=focaler_siou, iou_aux=nwd; iou_family=focaler, iou_variant=siou, slide_auto_iou_min=0.200, slide_delta=0.100, ema_decay=0.999, ema_tau=2000.0, iou_aux_ratio=0.500, focaler_d=0.000, focaler_u=0.950
```

4) Varifocal + MPDIoU（覆盖 varifocal、mpd 家族）：

```text
LossConfig cls_loss=varifocal, iou_loss=mpdiou, iou_aux=none; iou_family=mpd, iou_variant=mpdiou, varifocal_alpha=0.750, varifocal_gamma=2.000
```

5) QualityFocal + Wise Inner（覆盖 qualityfocal、wise inner、wise monotonic）：

```text
LossConfig cls_loss=qualityfocal, iou_loss=wiseiou_inner_diou, iou_aux=none; iou_family=wise, iou_variant=diou, qfl_beta=2.000, inner_iou_ratio=0.700, wiseiou_monotonous=True
```

6) Focal + Wise Focaler MPDIoU（覆盖 focal、wise focaler、wise+mpdiou）：

```text
LossConfig cls_loss=focal, iou_loss=wiseiou_focaler_mpdiou, iou_aux=none; iou_family=wise, iou_variant=mpdiou, focal_gamma=1.500, focal_alpha=0.250, focaler_d=0.000, focaler_u=0.950, wiseiou_monotonous=False
```

## 10. 参考链接

### 10.1 分类损失论文

- Focal Loss：https://arxiv.org/abs/1708.02002
- Varifocal Loss：https://arxiv.org/abs/2008.13367
- Quality Focal Loss (GFL)：https://arxiv.org/abs/2006.04388

### 10.2 IoU 主损失与变体论文

- GIoU：https://arxiv.org/abs/1902.09630
- DIoU / CIoU：https://arxiv.org/abs/1911.08287
- EIoU (Focal and Efficient IoU)：https://arxiv.org/abs/2101.08158
- SIoU：https://arxiv.org/abs/2205.12740
- Shape-IoU：https://arxiv.org/abs/2312.17663
- Inner-IoU：https://arxiv.org/abs/2311.02877
- Focaler-IoU：https://arxiv.org/abs/2401.10525
- MPDIoU：https://arxiv.org/abs/2307.07662
- Wise-IoU：https://arxiv.org/abs/2301.10051
- PIoU / PIoU2 (Powerful-IoU)：https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640

### 10.3 IoU 辅助分支论文

- Normalized Gaussian Wasserstein Distance (NWD)：https://arxiv.org/abs/2110.13389
- Gaussian Combined Distance (GCD)：https://arxiv.org/pdf/2510.27649

### 10.4 实现与讲解参考（非论文）

- SlideLoss/EMASlideLoss 源码参考（YOLO-FaceV2）：https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py
- EMASlideLoss 思路讲解（Bilibili）：https://www.bilibili.com/video/BV1W14y1i79U/?vd_source=c8452371e7ca510979593165c8d7ac27

## 11. 常见错误

- `Unsupported iou_loss=...`
  - 原因：命名不符合上述规则，或变体拼写错误。
  - 处理：优先按 `inner_<base>` / `focaler_<base>` / `wiseiou_<...>` 规则检查。

- 训练结果突然变化很大
  - 原因：切换了默认 loss 组合。
  - 处理：先用默认组合复现实验，再逐项开启新 loss。
