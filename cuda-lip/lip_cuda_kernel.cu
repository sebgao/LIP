#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

/*
Below is the fast version.
*/

template <typename scalar_t>
__global__ void LIPForward(const int nthreads,
                              const scalar_t *bottom_feature, const scalar_t *bottom_weight,
                              const int batches, const int channels,
                              const int height, const int width,
                              const int kernel, const int stride,
                              scalar_t *top_data){
    int pooled_height = height/stride;
    int pooled_width = width/stride;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width;
      const scalar_t *offset_bottom_feature = bottom_feature + offset;
      const scalar_t *offset_bottom_weight = bottom_weight + offset;

      scalar_t output = 0;
      scalar_t sum = 0;

      for(int iy=0; iy<kernel; iy++){
        const int y_offset = ph*stride + iy - kernel/2;
        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel; ix++){
          const int x_offset = pw*stride + ix - kernel/2;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          const scalar_t weight = exp(offset_bottom_weight[offset]);
          
          output += offset_bottom_feature[offset] * weight;
          sum += weight;
        }
      }
      top_data[index] = output/sum;
    }
}



/*
Below is the numerical-stable version, which might be slower.
*/

// template <typename scalar_t>
// __global__ void LIPForward(const int nthreads,
//                               const scalar_t *bottom_feature, const scalar_t *bottom_weight,
//                               const int batches, const int channels,
//                               const int height, const int width,
//                               const int kernel, const int stride,
//                               scalar_t *top_data){
//     int pooled_height = height/stride;
//     int pooled_width = width/stride;
//     CUDA_1D_KERNEL_LOOP(index, nthreads) {
//       int pw = index % pooled_width;
//       int ph = (index / pooled_width) % pooled_height;
//       int c = (index / pooled_width / pooled_height) % channels;
//       int n = index / pooled_width / pooled_height / channels;

//       const int offset = (n * channels + c) * height * width;
//       const scalar_t *offset_bottom_feature = bottom_feature + offset;
//       const scalar_t *offset_bottom_weight = bottom_weight + offset;

//       scalar_t output = 0;
//       scalar_t sum = 0;
//       scalar_t _max = 0;
      
//       for(int iy=0; iy<kernel; iy++){
//         const int y_offset = ph*stride + iy - kernel/2;
//         if(y_offset >= height || y_offset < 0)continue;
//         for(int ix=0; ix<kernel; ix++){
//           const int x_offset = pw*stride + ix - kernel/2;
//           if(x_offset >= width || x_offset < 0)continue;
//           const int offset = y_offset*width + x_offset;
//           _max = max(_max, offset_bottom_weight[offset]);
//         }
//       }

//       _max -= 15.0;

//       for(int iy=0; iy<kernel; iy++){
//         const int y_offset = ph*stride + iy - kernel/2;
//         if(y_offset >= height || y_offset < 0)continue;

//         for(int ix=0; ix<kernel; ix++){
//           const int x_offset = pw*stride + ix - kernel/2;
//           if(x_offset >= width || x_offset < 0)continue;
//           const int offset = y_offset*width + x_offset;

//           const scalar_t weight = exp(offset_bottom_weight[offset]-_max);
          
//           output += offset_bottom_feature[offset] * weight;
//           sum += weight;
//         }
//       }
//       top_data[index] = output/sum;
//     }
// }



int LIPForwardLaucher(const at::Tensor features, const at::Tensor weights,
  const int batches, const int channels,
  const int height, const int width,
  const int kernel, const int stride,
  at::Tensor output){
    const int output_size = batches * height/stride * width/stride * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "LIPLaucherForward", ([&] {
        const scalar_t *bottom_feature = features.data<scalar_t>();
        const scalar_t *bottom_weight = weights.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        LIPForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_feature, bottom_weight,

          batches, channels, height, width,
          kernel, stride,
          top_data);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}


template <typename scalar_t>
__global__ void LIPBackward(const int nthreads,
                              const scalar_t *diff_top, const scalar_t *data_top,
                              const scalar_t *data_feature, const scalar_t *data_weight,
                              const int batches, const int channels,
                              const int height, const int width,
                              const int kernel, const int stride,
                              scalar_t *diff_feature, scalar_t *diff_weight){
    int pooled_height = height/stride;
    int pooled_width = width/stride;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width;
      const scalar_t *offset_data_feature = data_feature + offset0;
      const scalar_t *offset_data_weight = data_weight + offset0;

      const scalar_t data_top_index = data_top[index];
      const scalar_t diff_top_index = diff_top[index];

      scalar_t *offset_diff_feature = diff_feature + offset0;
      scalar_t *offset_diff_weight = diff_weight + offset0;

      const int base_y = ph*stride - kernel/2;
      const int base_x = pw*stride - kernel/2;

      scalar_t sum_w = .0;
      for(int iy=0; iy<kernel; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;
            sum_w += exp(offset_data_weight[offset]);
        }
      }

      for(int iy=0; iy<kernel; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;
            
            const scalar_t grad_x = exp(offset_data_weight[offset])/(sum_w);
            const scalar_t grad_w = (offset_data_feature[offset]-data_top_index)*grad_x;
            atomicAdd(offset_diff_weight+offset, diff_top_index*grad_w);
            atomicAdd(offset_diff_feature+offset, diff_top_index*grad_x);
        }
      }
    }
}


int LIPBackwardLaucher(const at::Tensor top_grad, const at::Tensor top,
  const at::Tensor features, const at::Tensor weights,
  const int batches, const int channels,
  const int height, const int width,
  const int kernel, const int stride,
  at::Tensor d_features, at::Tensor d_weights){

    const int output_size = batches * height/stride * width/stride * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "LIPLaucherBackward", ([&] {
        scalar_t *diff_feature = d_features.data<scalar_t>();
        scalar_t *diff_weight = d_weights.data<scalar_t>();
        const scalar_t *data_top = top.data<scalar_t>();
        const scalar_t *diff_top = top_grad.data<scalar_t>(); 
        const scalar_t *data_feature = features.data<scalar_t>();
        const scalar_t *data_weight = weights.data<scalar_t>();

        LIPBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_top, data_top,
          data_feature, data_weight,
          batches, channels, height, width,
          kernel, stride,
          diff_feature, diff_weight);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}