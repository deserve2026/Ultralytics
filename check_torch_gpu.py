import warnings
warnings.filterwarnings('ignore')
from ultralytics.plugins.torch_utils import check_cuda

if __name__ == '__main__':
    check_cuda()