import timm
import torch
from thop import clever_format, profile

# print(timm.list_models())
# print(timm.list_models('*mobilenet*'))
print(timm.list_models("*repvit*"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 3, 640, 640).to(device)

model = timm.create_model("repvit_m0_9", pretrained=False, features_only=True)
model.to(device)
model.eval()

print(model.feature_info.channels())
for idx, feature in enumerate(model(dummy_input)):
    print(f"index:{idx} feature-size:{feature.size()}")

flops, params = profile(model.to(device), (dummy_input,), verbose=False)
flops, params = clever_format([flops * 2, params], "%.3f")
print(f"Total FLOPS: {flops}")
print(f"Total params: {params}")
