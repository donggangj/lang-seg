import torch
from torch import nn
from torch.onnx import export as ex_to_onnx


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokens = torch.zeros((1, 1, 1024))

    def forward(self, x):
        tokens = self.tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((tokens, x), dim=1)
        return x.transpose(1, 2)


def main():
    # test if torch==1.12.1 would work
    model = torch.jit.script(TestModule())
    dummy_input = torch.randn((1, 300, 1024))
    out = model(dummy_input)
    ex_to_onnx(torch.jit.script(model),
               dummy_input,
               'test.onnx',
               export_params=True,
               opset_version=14,
               do_constant_folding=True,
               input_names=['feature_in'],
               output_names=['feature_out'],
               dynamic_axes={'feature_in': {0: 'batch_sz'},
                             'feature_out': [0]},
               verbose=True)


if __name__ == '__main__':
    main()
