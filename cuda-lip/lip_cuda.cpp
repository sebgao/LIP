#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

int LIPForwardLaucher(const at::Tensor features, const at::Tensor weights,
                            const int batches, const int channels,
                            const int height, const int width,
                            const int kernel, const int stride,
                            at::Tensor output);

int LIPBackwardLaucher(const at::Tensor top_grad, const at::Tensor top,
                            const at::Tensor features, const at::Tensor weights,
                            const int batches, const int channels,
                            const int height, const int width,
                            const int kernel, const int stride,
                            at::Tensor d_features, at::Tensor d_weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int lip_forward_cuda(at::Tensor features, at::Tensor weights,
                        const int kernel, const int stride,
                        at::Tensor output) {
    CHECK_INPUT(features);
    CHECK_INPUT(weights);
    CHECK_INPUT(output);

    int batches = features.size(0);
    int channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);

    LIPForwardLaucher(features, weights,
                        batches, channels, height, width,
                        kernel, stride,
                        output);
    return 1;
}

int lip_backward_cuda(const at::Tensor top_grad, const at::Tensor top,
                        const at::Tensor features, const at::Tensor weights,
                        const int kernel, const int stride,
                        at::Tensor d_features, at::Tensor d_weights) {
    CHECK_INPUT(top_grad);
    CHECK_INPUT(top);
    CHECK_INPUT(features);
    CHECK_INPUT(weights);

    CHECK_INPUT(d_features);
    CHECK_INPUT(d_weights);

    int batches = d_features.size(0);
    int channels = d_features.size(1);
    int height = d_features.size(2);
    int width = d_features.size(3);

    LIPBackwardLaucher(top_grad, top,
                        features, weights,
                        batches, channels, height, width,
                        kernel, stride,
                        d_features, d_weights);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lip_forward_cuda, "LIP forward (CUDA)");
  m.def("backward", &lip_backward_cuda, "LIP backward (CUDA)");
}
