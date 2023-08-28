import onnx
import onnxruntime
from model import Model
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(onnx.__version__)
    print(onnxruntime.__version__)
    print(torch.__version__)
    pth = 'shufflenet_v2_x1_0.pt'
    source = 'shufflenet_v2_x1_0.onnx'

    model = Model(cfg='s.yaml', ch=3, nc=25, auc=False)
    model.load_state_dict(torch.load(pth, map_location='cpu'))
    model = model.eval().to(device)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            source,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
        )

    onnx_model = onnx.load(source)
    onnx.checker.check_model(onnx_model)
    print('保存成功')


if __name__ == '__main__':
    main()
