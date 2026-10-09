import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys, copy
torch.autograd.set_detect_anomaly(True)
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose, Pose26
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import LOGGER
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, BOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[1m", "\033[0m"

def patch_pose_classes_for_gradcam():
    """修复 Pose 和 Pose26 类使其兼容 Grad-CAM，移除 inplace 操作"""
    
    # 修复 Pose 类
    def pose_kpts_decode_no_inplace(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions (no inplace operations)."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                # 强制使用非 inplace 操作
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y
    
    # 修复 Pose26 类
    def pose26_kpts_decode_no_inplace(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions (no inplace operations)."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                # 强制使用非 inplace 操作
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y
    
    # 应用补丁
    Pose.kpts_decode = pose_kpts_decode_no_inplace
    Pose26.kpts_decode = pose26_kpts_decode_no_inplace

patch_pose_classes_for_gradcam()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (top, bottom, left, right)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][0][:, 4:4 + self.model.nc]
            boxes_ = result[0][0][:, :4]
            mask_p, mask_nm = result[0][1].squeeze(), result[0][0][:, 4 + self.model.nc:].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    @staticmethod
    def _accumulate(acc, value):
        return value if acc is None else acc + value

    @staticmethod
    def _zero_scalar_like(tensor):
        # Keep the zero target connected to autograd graph so Grad-CAM layers receive zero (not None) gradients.
        return tensor.sum() * 0.0
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        acc = None
        loop_count = min(int(post_result.size(0) * self.ratio), post_result.size(0))
        for i in trange(loop_count):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type in ("class", "all"):
                acc = self._accumulate(acc, post_result[i, 0] if self.end2end else post_result[i].max())
            if self.ouput_type in ("box", "all"):
                for j in range(4):
                    acc = self._accumulate(acc, pre_post_boxes[i, j])
        return acc if acc is not None else self._zero_scalar_like(post_result)

class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        acc = None
        loop_count = min(int(post_result.size(0) * self.ratio), post_result.size(0))
        for i in trange(loop_count):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ("class", "all"):
                acc = self._accumulate(acc, post_result[i].max())
            if self.ouput_type in ("box", "all"):
                for j in range(4):
                    acc = self._accumulate(acc, pre_post_boxes[i, j])
            if self.ouput_type in ("segment", "all"):
                acc = self._accumulate(acc, pre_post_mask[i].mean())
        return acc if acc is not None else self._zero_scalar_like(post_result)

class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        acc = None
        loop_count = min(int(post_result.size(0) * self.ratio), post_result.size(0))
        for i in trange(loop_count):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ("class", "all"):
                acc = self._accumulate(acc, post_result[i].max())
            if self.ouput_type in ("box", "all"):
                for j in range(4):
                    acc = self._accumulate(acc, pre_post_boxes[i, j])
            if self.ouput_type in ("pose", "all"):
                acc = self._accumulate(acc, pre_post_pose[i].mean())
        return acc if acc is not None else self._zero_scalar_like(post_result)

class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        acc = None
        loop_count = min(int(post_result.size(0) * self.ratio), post_result.size(0))
        for i in trange(loop_count):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ("class", "all"):
                acc = self._accumulate(acc, post_result[i].max())
            if self.ouput_type in ("box", "all"):
                for j in range(4):
                    acc = self._accumulate(acc, pre_post_boxes[i, j])
            if self.ouput_type in ("obb", "all"):
                acc = self._accumulate(acc, pre_post_angle[i])
        return acc if acc is not None else self._zero_scalar_like(post_result)

class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        return data.max()

class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size, letterbox_auto):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        LOGGER.info(f'{ORANGE}model class info:{model_names}{RESET}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False
        if model.end2end:
            model.end2end = False
        
        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")
        
        target_layers = [model.model[l] for l in layer]
        cam_methods = {
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "GradCAM": GradCAM,
            "XGradCAM": XGradCAM,
            "EigenCAM": EigenCAM,
            "HiResCAM": HiResCAM,
            "LayerCAM": LayerCAM,
            "RandomCAM": RandomCAM,
            "EigenGradCAM": EigenGradCAM,
            "KPCA_CAM": KPCA_CAM,
            "AblationCAM": AblationCAM,
        }
        if method not in cam_methods:
            raise ValueError(f"Unsupported CAM method '{method}'. Available methods: {', '.join(cam_methods)}")
        method = cam_methods[method](model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2) # 绘制检测框
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)  # 绘制类别、置信度
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path, save_path):
        # img process
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            LOGGER.error(f"{RED}{img_path} read failure.{RESET}")
            return False
        if img is None:
            LOGGER.error(f"{RED}{img_path} decode failure (not an image or corrupted file).{RESET}")
            return False
        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=self.letterbox_auto)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        LOGGER.info(f'{BOLD}{ORANGE}tensor size:{tensor.size()}{RESET}')
        
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError:
            LOGGER.warning(f"{CYAN}self.method(tensor, [self.target]) failure.{RESET}")
            return False
        
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7, verbose=False)[0]
        if self.renormalize and self.task in ['detect', 'segment', 'pose']:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
        if self.show_result:
            cam_image = pred.plot(img=cam_image,
                                  conf=True, # 显示置信度
                                  font_size=None, # 字体大小，None为根据当前image尺寸计算
                                  line_width=None, # 线条宽度，None为根据当前image尺寸计算
                                  labels=False, # 显示标签
                                  )
        
        # 去掉padding边界
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
        return True
    
    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            success, failed = 0, 0
            for img_path_ in os.listdir(img_path):
                ok = self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
                success += int(ok)
                failed += int(not ok)
            LOGGER.info(f"{BOLD}{ORANGE}processed images: success={success}, failed={failed}{RESET}")
        else:
            ok = self.process(img_path, f'{save_path}/result.png')
            if not ok:
                LOGGER.error(f"{RED}failed to process input image: {img_path}{RESET}")
        
        LOGGER.info(f'{BOLD}{MAGENTA}进度条不满是正常现象,只要进度条不是0,都可以进行出图.{RESET}')
        
def get_params():
    params = {
        'weight': 'yolo26n.pt', # 现在只需要指定权重即可,不需要指定cfg
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
        'layer': [16, 19, 22],
        'backward_type': 'all', # detect:<class, box, all> segment:<class, box, segment, all> pose:<box, keypoint, all> obb:<box, angle, all> classify:<all>
        'conf_threshold': 0.2, # 0.2
        'ratio': 0.02, # 0.02-0.1
        'show_result': True, # 不需要绘制结果请设置为False
        'renormalize': False, # 需要把热力图限制在框内请设置为True(仅对detect,segment,pose有效)
        'task':'detect', # 任务(detect,segment,pose,obb,classify)
        'img_size':640, # 图像尺寸
        'letterbox_auto': True # 如果需要固定成宽高一样就设置为False，部分改进可能需要输入的宽高一致，不然会报错
    }
    return params

# pip install grad-cam==1.5.5 --no-deps
if __name__ == '__main__':
    model = yolo_heatmap(**get_params())
    model(r'/root/dataset/coco/images/val2017/000000361238.jpg', 'heatmap_result')
    # model(r'/root/dataset/coco/images/val2017', 'heatmap_result')
    # model(r'/root/code/project/datasets/DOTAv1.5/images/test', 'heatmap_result')