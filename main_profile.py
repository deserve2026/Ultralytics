import os, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定使用第一张显卡
warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.plugins.utils import suppress_logging

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, BOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[1m", "\033[0m"

if __name__ == '__main__':
    yaml_path = 'ultralytics/cfg/models/26/yolo26n.yaml' # 选择你的yaml
    imgsz = [640, 640] # 顺序为H、W

    model = YOLO(yaml_path, verbose=True)
    with suppress_logging():
        model.fuse()
    print(f'{BOLD}{RED}-------------------------- fused 代表的是重参数化后的模型 --------------------------{RESET}')

    p = next(model.model.parameters())
    inputs = torch.randn((1, 3, *imgsz), device=p.device)
    model.model.eval()
    # Prefer inference_mode for lower overhead; fall back to no_grad for older torch versions.
    inference_ctx = getattr(torch, "inference_mode", None)
    context_manager = inference_ctx if callable(inference_ctx) else torch.no_grad
    with context_manager():
        model.model.predict(inputs, profile=True)

    model.info(detailed=False, imgsz=imgsz)
