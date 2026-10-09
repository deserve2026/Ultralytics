import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

# pip install onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=17,
        half=False,
        verbose=True,
        # end2end=False # 如果训练的是NMSFree类型的模型，不想用一对一的头可以设置False
    )
