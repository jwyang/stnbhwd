#include "utils.h"
#include <stdio.h>
// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getTopLeft(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   float xcoord = (x + 1) * (width - 1) / 2;
   point = floor(xcoord);
   weight = 1 - (xcoord - point);
}

__device__ void getBottomRight(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   float xcoord = (x + 1) * (width - 1) / 2;
   point = ceil(xcoord);
   weight = 1 - (point - xcoord);
}

__device__ bool between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

__device__ void sumReduceShMem(volatile float s[])
{
   /* obviously only works for 32 elements */
   /* sums up a shared memory array of 32 elements, stores it in s[0] */
   /* whole warp can then read first element (broadcasting) */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}



__global__ void bilinearSamplingFromGrid(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* masks_data, int masks_strideBatch, int masks_strideYX, int masks_strideHeight, int masks_strideWidth,
                                         float* canvas_data, int canvas_strideBatch, int canvas_strideYX, int canvas_strideHeight, int canvas_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)

   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < output_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < output_width;
   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;

   const int b = blockIdx.z;

   float yf,xf;

   __shared__ float gridData[32];
   if (threadIdx.y==0 && withinGridBounds)
   {
     // #if __CUDA_ARCH__>=200
     //    printf("%d \n", grids_strideWidth);
     // #endif
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + threadIdx.x];
   }
   __syncthreads();
   if(!withinImageBounds) return;
   yf = gridData[threadIdx.y*2];
   xf = gridData[threadIdx.y*2+1];

   int yInTopLeft, xInTopLeft;
   float yWeightTopLeft, xWeightTopLeft;
   getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
   getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

   // xWeightTopLeft = 0.5;
   // yWeightTopLeft = 0.5;

   const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;

   const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
   const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
   const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
   const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

   const int inTopLeftMaskAddress = masks_strideBatch * b + masks_strideHeight * yInTopLeft + masks_strideWidth * xInTopLeft;
   const int inTopRightMaskAddress = inTopLeftMaskAddress + masks_strideWidth;
   const int inBottomLeftMaskAddress = inTopLeftMaskAddress + masks_strideHeight;
   const int inBottomRightMaskAddress = inBottomLeftMaskAddress + masks_strideWidth;

   float v=0;
   float inTopLeft=0;
   float inTopRight=0;
   float inBottomLeft=0;
   float inBottomRight=0;

   float m = 0;
   float inTopLeftMask=0;
   float inTopRightMask=0;
   float inBottomLeftMask=0;
   float inBottomRightMask=0;

   bool topLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft, 0, height-1);
   bool topRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft, 0, height-1);
   bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft+1, 0, height-1);
   bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft+1, 0, height-1);

   if(topLeftIsIn) inTopLeftMask = masks_data[inTopLeftMaskAddress];
   if(topRightIsIn) inTopRightMask = masks_data[inTopRightMaskAddress];
   if(bottomLeftIsIn) inBottomLeftMask = masks_data[inBottomLeftMaskAddress];
   if(bottomRightIsIn) inBottomRightMask = masks_data[inBottomRightMaskAddress];

   m = xWeightTopLeft * yWeightTopLeft * inTopLeftMask
     + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRightMask
     + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeftMask
     + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRightMask;

   // interpolation happens here
   for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
   {
      // jw2yang: do not change output_data when it locates outside the source image,
      // Todo: check backward after considering this case.
      if (!topLeftIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn)
        output_data[outAddress + t] = canvas_data[outAddress + t];

      if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
      if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
      if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
      if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

      v = xWeightTopLeft * yWeightTopLeft * inTopLeft
        + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
        + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
        + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

      // we do not replace the canvas region with foreground, instead, we add value together.
      output_data[outAddress + t] = (1 - m) * canvas_data[outAddress + t] + m * v;
      // output_data[outAddress + t] = v;
   }
}

static int cunn_BilinearSamplerBHWD_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *masks = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *canvas = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");


   dim3 blocks((output->size[2]+15)/16, output->size[1], output->size[0]);
   dim3 threads(32,16);

   /* assume BHWD */
   bilinearSamplingFromGrid <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (THCudaTensor_data(state, inputImages),
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      THCudaTensor_data(state, grids),
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_data(state, masks),
                                                      THCudaTensor_stride(state, masks, 0),
                                                      THCudaTensor_stride(state, masks, 3),
                                                      THCudaTensor_stride(state, masks, 1),
                                                      THCudaTensor_stride(state, masks, 2),
                                                      THCudaTensor_data(state, canvas),
                                                      THCudaTensor_stride(state, canvas, 0),
                                                      THCudaTensor_stride(state, canvas, 3),
                                                      THCudaTensor_stride(state, canvas, 1),
                                                      THCudaTensor_stride(state, canvas, 2),
                                                      THCudaTensor_data(state, output),
                                                      THCudaTensor_stride(state, output, 0),
                                                      THCudaTensor_stride(state, output, 3),
                                                      THCudaTensor_stride(state, output, 1),
                                                      THCudaTensor_stride(state, output, 2),
                                                      THCudaTensor_size(state, inputImages, 3),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2),
                                                      THCudaTensor_size(state, output, 2));


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

template<bool onlyGrid> __global__ void backwardBilinearSampling(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYX, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* masks_data, int masks_strideBatch, int masks_strideYX, int masks_strideHeight, int masks_strideWidth,
                                         float* gradMasks_data, int gradMasks_strideBatch, int gradMasks_strideYX, int gradMasks_strideHeight, int gradMasks_strideWidth,
                                         float* canvas_data, int canvas_strideBatch, int canvas_strideYX, int canvas_strideHeight, int canvas_strideWidth,
                                         float* gradCanvas_data, int gradCanvas_strideBatch, int gradCanvas_strideYX, int gradCanvas_strideHeight, int gradCanvas_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int gradOutput_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates
   // z = batch index
   // threads : used for features

   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < gradOutput_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < gradOutput_width;

   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;

   const int b = blockIdx.z;

   float yf,xf;

   __shared__ float gridData[32];
   if (threadIdx.y==0 && withinGridBounds)
   {
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + threadIdx.x];
   }
   __syncthreads();

   if(withinImageBounds)
   {
      yf = gridData[threadIdx.y*2];
      xf = gridData[threadIdx.y*2+1];

      int yInTopLeft, xInTopLeft;
      float yWeightTopLeft, xWeightTopLeft;
      getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
      getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

      // xWeightTopLeft = 0.5;
      // yWeightTopLeft = 0.5;

      const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
      const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
      const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
      const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

      const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
      const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
      const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
      const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

      const int inTopLeftMaskAddress = masks_strideBatch * b + masks_strideHeight * yInTopLeft + masks_strideWidth * xInTopLeft;
      const int inTopRightMaskAddress = inTopLeftMaskAddress + masks_strideWidth;
      const int inBottomLeftMaskAddress = inTopLeftMaskAddress + masks_strideHeight;
      const int inBottomRightMaskAddress = inBottomLeftMaskAddress + masks_strideWidth;

      const int gradMasksTopLeftAddress = gradMasks_strideBatch * b + gradMasks_strideHeight * yInTopLeft + gradMasks_strideWidth * xInTopLeft;
      const int gradMasksTopRightAddress = gradMasksTopLeftAddress + gradMasks_strideWidth;
      const int gradMasksBottomLeftAddress = gradMasksTopLeftAddress + gradMasks_strideHeight;
      const int gradMasksBottomRightAddress = gradMasksBottomLeftAddress + gradMasks_strideWidth;

      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

      float topLeftDotProduct = 0;
      float topRightDotProduct = 0;
      float bottomLeftDotProduct = 0;
      float bottomRightDotProduct = 0;

      bool topLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft, 0, height-1);
      bool topRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft, 0, height-1);
      bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft+1, 0, height-1);
      bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft+1, 0, height-1);

      float v = 0;
      float inTopLeft=0;
      float inTopRight=0;
      float inBottomLeft=0;
      float inBottomRight=0;

      float c = 0;

      float m = 0;
      float inTopLeftMask=0;
      float inTopRightMask=0;
      float inBottomLeftMask=0;
      float inBottomRightMask=0;

      if(topLeftIsIn) inTopLeftMask = masks_data[inTopLeftMaskAddress];
      if(topRightIsIn) inTopRightMask = masks_data[inTopRightMaskAddress];
      if(bottomLeftIsIn) inBottomLeftMask = masks_data[inBottomLeftMaskAddress];
      if(bottomRightIsIn) inBottomRightMask = masks_data[inBottomRightMaskAddress];

      m = xWeightTopLeft * yWeightTopLeft * inTopLeftMask
        + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRightMask
        + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeftMask
        + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRightMask;

      /*
         In that loop we accumulate
         - gradients into the gradInputImages array with atomic adds
         - we compute the dot product that we need for the grid gradient
      */

      for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
      {

         float gradOutValue = gradOutput_data[gradOutputAddress + t];
         float gradOutValue_fg = gradOutValue * m;
         float gradOutValue_bg = gradOutValue * (1 - m);
         // bool between(int value, int lowerBound, int upperBound)
         if(topLeftIsIn)
         {
            float inTopLeft = inputImages_data[inTopLeftAddress + t];
            topLeftDotProduct += inTopLeft * gradOutValue_fg;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftAddress + t], xWeightTopLeft * yWeightTopLeft * gradOutValue_fg);
         }

         if(topRightIsIn)
         {
            float inTopRight = inputImages_data[inTopRightAddress + t];
            topRightDotProduct += inTopRight * gradOutValue_fg;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightAddress + t], (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue_fg);
         }

         if(bottomLeftIsIn)
         {
            float inBottomLeft = inputImages_data[inBottomLeftAddress + t];
            bottomLeftDotProduct += inBottomLeft * gradOutValue_fg;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftAddress + t], xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue_fg);
         }

         if(bottomRightIsIn)
         {
            float inBottomRight = inputImages_data[inBottomRightAddress + t];
            bottomRightDotProduct += inBottomRight * gradOutValue_fg;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightAddress + t], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue_fg);
         }

         // jw2yang: copy the gradients outside the object region to canvas, and inside region
         if (!topLeftIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn) {
            gradCanvas_data[gradOutputAddress + t] = gradOutValue;
         }
         else {
            gradCanvas_data[gradOutputAddress + t] = gradOutValue_bg;
         }

         // jw2yang: compute the gradient mask value
         if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
         if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
         if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
         if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];
         v = xWeightTopLeft * yWeightTopLeft * inTopLeft
           + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
           + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
           + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

         c = canvas_data[gradOutputAddress + t];

         float gradMaskValue = gradOutValue * (v - c);


         // update gradient on mask map
         if(topLeftIsIn)
         {
            if(!onlyGrid) atomicAdd(&gradMasks_data[gradMasksTopLeftAddress], xWeightTopLeft * yWeightTopLeft * gradMaskValue);
         }

         if(topRightIsIn)
         {
            if(!onlyGrid) atomicAdd(&gradMasks_data[gradMasksTopRightAddress], (1 - xWeightTopLeft) * yWeightTopLeft * gradMaskValue);
         }

         if(bottomLeftIsIn)
         {
            if(!onlyGrid) atomicAdd(&gradMasks_data[gradMasksBottomLeftAddress], xWeightTopLeft * (1 - yWeightTopLeft) * gradMaskValue);
         }

         if(bottomRightIsIn)
         {
            if(!onlyGrid) atomicAdd(&gradMasks_data[gradMasksBottomRightAddress], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradMaskValue);
         }
      }
      /*
         Here we reduce the dot product and compute the grid gradient before writing it.
      */

      /* could do shuffles and use no shmem at all but cuda arch is 2.0 */
      __shared__ volatile float __shmem[16][32];
      __shmem[threadIdx.y][threadIdx.x] = topLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topLeftDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = topRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topRightDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = bottomLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomLeftDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = bottomRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomRightDotProduct = __shmem[threadIdx.y][0];

      yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
      xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

      if(threadIdx.x==0)
      {
         gridData[threadIdx.y*2] = yf * (inputImages_height-1) / 2;
         gridData[threadIdx.y*2+1] = xf * (inputImages_width-1) / 2;
      }
   }// must put a big if condition in order not to hang at __syncthreads()...
   __syncthreads();

   if(threadIdx.y==0 && withinGridBounds)
       gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + threadIdx.x] = gridData[threadIdx.x];
}





static int cunn_BilinearSamplerBHWD_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *masks = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *canvas = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");
  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 7, "torch.CudaTensor");
  THCudaTensor *gradMasks = (THCudaTensor *)luaT_checkudata(L, 8, "torch.CudaTensor");
  THCudaTensor *gradCanvas = (THCudaTensor *)luaT_checkudata(L, 9, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 10, "torch.CudaTensor");

   dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 threads(32,16);

   backwardBilinearSampling <false> <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (
                                                      THCudaTensor_data(state, inputImages),
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      THCudaTensor_data(state, gradInputImages),
                                                      THCudaTensor_stride(state, gradInputImages, 0),
                                                      THCudaTensor_stride(state, gradInputImages, 3),
                                                      THCudaTensor_stride(state, gradInputImages, 1),
                                                      THCudaTensor_stride(state, gradInputImages, 2),
                                                      THCudaTensor_data(state, grids),
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_data(state, gradGrids),
                                                      THCudaTensor_stride(state, gradGrids, 0),
                                                      THCudaTensor_stride(state, gradGrids, 3),
                                                      THCudaTensor_stride(state, gradGrids, 1),
                                                      THCudaTensor_stride(state, gradGrids, 2),
                                                      THCudaTensor_data(state, masks),
                                                      THCudaTensor_stride(state, masks, 0),
                                                      THCudaTensor_stride(state, masks, 3),
                                                      THCudaTensor_stride(state, masks, 1),
                                                      THCudaTensor_stride(state, masks, 2),
                                                      THCudaTensor_data(state, gradMasks),
                                                      THCudaTensor_stride(state, gradMasks, 0),
                                                      THCudaTensor_stride(state, gradMasks, 3),
                                                      THCudaTensor_stride(state, gradMasks, 1),
                                                      THCudaTensor_stride(state, gradMasks, 2),
                                                      THCudaTensor_data(state, canvas),
                                                      THCudaTensor_stride(state, canvas, 0),
                                                      THCudaTensor_stride(state, canvas, 3),
                                                      THCudaTensor_stride(state, canvas, 1),
                                                      THCudaTensor_stride(state, canvas, 2),
                                                      THCudaTensor_data(state, gradCanvas),
                                                      THCudaTensor_stride(state, gradCanvas, 0),
                                                      THCudaTensor_stride(state, gradCanvas, 3),
                                                      THCudaTensor_stride(state, gradCanvas, 1),
                                                      THCudaTensor_stride(state, gradCanvas, 2),
                                                      THCudaTensor_data(state, gradOutput),
                                                      THCudaTensor_stride(state, gradOutput, 0),
                                                      THCudaTensor_stride(state, gradOutput, 3),
                                                      THCudaTensor_stride(state, gradOutput, 1),
                                                      THCudaTensor_stride(state, gradOutput, 2),
                                                      THCudaTensor_size(state, inputImages, 3),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2),
                                                      THCudaTensor_size(state, gradOutput, 2));



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

// Added by Jianwei Yang, do subsampling for images
__global__ void subSamplingFromGrid(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* masks_data, int masks_strideBatch, int masks_strideYX, int masks_strideHeight, int masks_strideWidth,
                                         float* canvas_data, int canvas_strideBatch, int canvas_strideYX, int canvas_strideHeight, int canvas_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_height, int output_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)

   // blocks.x: width / 16
   // blocks.y: height
   // blocks.z: batchSize
   // blockDim.x.y.z: number of threads in each direction, here blockDim.y = 16
   const int xOut = blockIdx.x*blockDim.y+threadIdx.y; // compute the x coordinate of feature map
   const int yOut = blockIdx.y;
   const int b = blockIdx.z;

   const bool withinImageBounds = xOut < output_width; // check whether x exceed the boundary
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < output_width; // check whether

   if(!withinImageBounds || !withinGridBounds) return;

   const int width = inputImages_width;
   const int height = inputImages_height;

   int xOut_l = xOut > 0 ? (xOut - 1) : xOut;
   int yOut_t = yOut > 0 ? (yOut - 1) : yOut;
   int xOut_r = xOut < (output_width - 1) ? (xOut + 1) : xOut;
   int yOut_b = yOut < (output_height - 1) ? (yOut + 1) : yOut;

   float yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
   float xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

   // get coordinates for four corners of target response map in the source response map
   float yf_tl = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_l*grids_strideWidth];
   float xf_tl = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_l*grids_strideWidth + 1];

   float yf_tr = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_r*grids_strideWidth];
   float xf_tr = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_r*grids_strideWidth + 1];

   float yf_bl = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_l*grids_strideWidth];
   float xf_bl = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_l*grids_strideWidth + 1];

   float yf_br = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_r*grids_strideWidth];
   float xf_br = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_r*grids_strideWidth + 1];

   // compute the bottom left coordinates of source response map
   float xs_tl = min(min(min(xf_tl, xf_tr), xf_bl), xf_br);
   float ys_tl = min(min(min(yf_tl, yf_tr), yf_bl), yf_br);

   float xs_br = max(max(max(xf_tl, xf_tr), xf_bl), xf_br);
   float ys_br = max(max(max(yf_tl, yf_tr), yf_bl), yf_br);

   int yi_t, xi_l;
   float yWeight_t, xWeight_l;

   int yi_b, xi_r;
   float yWeight_b, xWeight_r;

   // compute the top left, top right, bottom left, bottom right corner for current grid
   // compute the nearest bottom right coordiate in source map of top left grid coordiate
   getBottomRight(xs_tl, inputImages_width, xi_l, xWeight_l);
   getBottomRight(ys_tl, inputImages_height, yi_t, yWeight_t);
   // compute the nearest top left coordiate in source map of bottom right grid coordiate
   getTopLeft(xs_br, inputImages_width, xi_r, xWeight_r);
   getTopLeft(ys_br, inputImages_height, yi_b, yWeight_b);

   bool topLeftIsIn = between(xi_l, 0, width-1) && between(yi_t, 0, height-1);
   bool topRightIsIn = between(xi_r, 0, width-1) && between(yi_t, 0, height-1);
   bool bottomLeftIsIn = between(xi_l, 0, width-1) && between(yi_b, 0, height-1);
   bool bottomRightIsIn = between(xi_r, 0, width-1) && between(yi_b, 0, height-1);

   const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;

   if (!topRightIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn) {
     for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
     {
       output_data[outAddress + t] = canvas_data[outAddress + t];
     }
   } else {
     // compute average mask value in region [topleft, bottom right]
     float weights[32];
     float m = 0;
     float weight_sum = 0;
     int id_point = 0;
     for (int y = yi_t; y <= yi_b; ++y) {
       if (!between(y, 0, height-1)) continue;
       for (int x = xi_l; x <= xi_r; ++x) {
         if (!between(x, 0, width-1)) continue;
         float weight = __expf((x - xf) * (x - xf) + (y - yf) * (y - yf));
         weight_sum += weight;
         weights[id_point] = weight;
         int address = masks_strideBatch * b + masks_strideHeight * y + masks_strideWidth * x;
         m += weight * masks_data[address];
         ++id_point;
       }
     }

     #if __CUDA_ARCH__>=200
        printf("%f ", weight_sum);
     #endif

     m /= weight_sum;

     float v=0;
     // interpolation happens here
     // compute the address (location) for output

     for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
     {
        // jw2yang: do not change output_data when it locates outside the source image,
        // Todo: check backward after considering this case.
        if (!topLeftIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn)
          output_data[outAddress + t] = canvas_data[outAddress + t];

        v = 0;
        id_point = 0;
        for (int y = yi_t; y <= yi_b; ++y) {
          if (!between(y, 0, height-1)) continue;
          for (int x = xi_l; x <= xi_r; ++x) {
            if (!between(x, 0, width-1)) continue;
            int address = inputImages_strideBatch * b + inputImages_strideHeight * y + inputImages_strideWidth * x + t;
            v += weights[id_point] * inputImages_data[address];
          }
        }
        v /= weight_sum;

        // we do not replace the canvas region with foreground, instead, we add value together.
        output_data[outAddress + t] = (1 - m) * canvas_data[outAddress + t] + m * v;
        // output_data[outAddress + t] = v;
     }
   }
}

static int cunn_SubSamplerBHWD_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *masks = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *canvas = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  // blocks.x: width / 16
  // blocks.y: height
  // blocks.z: batchSize
  dim3 blocks((output->size[2]+15)/16, output->size[1], output->size[0]);
  dim3 threads(32,16);

   /* assume BHWD */
   subSamplingFromGrid <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (THCudaTensor_data(state, inputImages),
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      THCudaTensor_data(state, grids),
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_data(state, masks),
                                                      THCudaTensor_stride(state, masks, 0),
                                                      THCudaTensor_stride(state, masks, 3),
                                                      THCudaTensor_stride(state, masks, 1),
                                                      THCudaTensor_stride(state, masks, 2),
                                                      THCudaTensor_data(state, canvas),
                                                      THCudaTensor_stride(state, canvas, 0),
                                                      THCudaTensor_stride(state, canvas, 3),
                                                      THCudaTensor_stride(state, canvas, 1),
                                                      THCudaTensor_stride(state, canvas, 2),
                                                      THCudaTensor_data(state, output),
                                                      THCudaTensor_stride(state, output, 0),
                                                      THCudaTensor_stride(state, output, 3),
                                                      THCudaTensor_stride(state, output, 1),
                                                      THCudaTensor_stride(state, output, 2),
                                                      THCudaTensor_size(state, inputImages, 3),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2),
                                                      THCudaTensor_size(state, output, 1),
                                                      THCudaTensor_size(state, output, 2));


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

template<bool onlyGrid> __global__ void backwardSubSampling(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYX, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* masks_data, int masks_strideBatch, int masks_strideYX, int masks_strideHeight, int masks_strideWidth,
                                         float* gradMasks_data, int gradMasks_strideBatch, int gradMasks_strideYX, int gradMasks_strideHeight, int gradMasks_strideWidth,
                                         float* canvas_data, int canvas_strideBatch, int canvas_strideYX, int canvas_strideHeight, int canvas_strideWidth,
                                         float* gradCanvas_data, int gradCanvas_strideBatch, int gradCanvas_strideYX, int gradCanvas_strideHeight, int gradCanvas_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int gradOutput_height, int gradOutput_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates
   // z = batch index
   // threads : used for features

   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;
   const int b = blockIdx.z;

   const bool withinImageBounds = xOut < gradOutput_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < gradOutput_width;

   if(withinImageBounds && withinGridBounds)
   {
     int xOut_l = xOut > 0 ? (xOut - 1) : xOut;
     int yOut_t = yOut > 0 ? (yOut - 1) : yOut;
     int xOut_r = xOut < (gradOutput_width - 1) ? (xOut + 1) : xOut;
     int yOut_b = yOut < (gradOutput_height - 1) ? (yOut + 1) : yOut;

     float yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
     float xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

     // get coordinates for four corners of target response map in the source response map
     float yf_tl = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_l*grids_strideWidth];
     float xf_tl = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_l*grids_strideWidth + 1];

     float yf_tr = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_r*grids_strideWidth];
     float xf_tr = grids_data[b*grids_strideBatch + yOut_t*grids_strideHeight + xOut_r*grids_strideWidth + 1];

     float yf_bl = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_l*grids_strideWidth];
     float xf_bl = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_l*grids_strideWidth + 1];

     float yf_br = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_r*grids_strideWidth];
     float xf_br = grids_data[b*grids_strideBatch + yOut_b*grids_strideHeight + xOut_r*grids_strideWidth + 1];

     // compute the bottom left coordinates of source response map
     float xs_tl = min(min(min(xf_tl, xf_tr), xf_bl), xf_br);
     float ys_tl = min(min(min(yf_tl, yf_tr), yf_bl), yf_br);

     float xs_br = max(max(max(xf_tl, xf_tr), xf_bl), xf_br);
     float ys_br = max(max(max(yf_tl, yf_tr), yf_bl), yf_br);

     int yi_t, xi_l;
     float yWeight_t, xWeight_l;

     int yi_b, xi_r;
     float yWeight_b, xWeight_r;

     // compute the top left, top right, bottom left, bottom right corner for current grid
     // compute the nearest bottom right coordiate in source map of top left grid coordiate
     getBottomRight(xs_tl, inputImages_width, xi_l, xWeight_l);
     getBottomRight(ys_tl, inputImages_height, yi_t, yWeight_t);
     // compute the nearest top left coordiate in source map of bottom right grid coordiate
     getTopLeft(xs_br, inputImages_width, xi_r, xWeight_r);
     getTopLeft(ys_br, inputImages_height, yi_b, yWeight_b);

     bool topLeftIsIn = between(xi_l, 0, width-1) && between(yi_t, 0, height-1);
     bool topRightIsIn = between(xi_r, 0, width-1) && between(yi_t, 0, height-1);
     bool bottomLeftIsIn = between(xi_l, 0, width-1) && between(yi_b, 0, height-1);
     bool bottomRightIsIn = between(xi_r, 0, width-1) && between(yi_b, 0, height-1);

     const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;
     if (!topLeftIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn) {
       for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
       {
         float gradOutValue = gradOutput_data[gradOutputAddress + t];
         gradCanvas_data[gradOutputAddress + t] = gradOutValue;
       }
     }
     else { // if there are some points inside the freground image and mask
       float weights[32];
       float bias_x[32];
       float bias_y[32];
       float bias_x_weighted = 0;
       float bias_y_weighted = 0;
       float m = 0;
       float weight_sum = 0;
       int id_point = 0;
       for (int y = yi_t; y <= yi_b; ++y) {
         if (!between(y, 0, height-1)) continue;
         for (int x = xi_l; x <= xi_r; ++x) {
           if (!between(x, 0, width-1)) continue;
           bias_x[id_point] = 2 * (xf - x); // multiply 2 since it is derivative of square
           bias_y[id_point] = 2 * (yf - y); // multiply 2 since it is derivative of square
           float weight = __expf(bias_x[id_point] * bias_x[id_point] + bias_y[id_point] * bias_y[id_point]);
           bias_x_weighted += weight * bias_x[id_point];
           bias_y_weighted += weight * bias_y[id_point];
           weight_sum += weight;
           weights[id_point] = weight;
           int address = masks_strideBatch * b + masks_strideHeight * y + masks_strideWidth * x;
           m += weight * masks_data[address];
           ++id_point;
         }
       }
       m /= weight_sum;
       float c = 0;
       float grad_xf = 0;
       float grad_yf = 0;
       for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
       {
         float gradOutValue = gradOutput_data[gradOutputAddress + t];
         float gradOutValue_fg = gradOutValue * m;
         float gradOutValue_bg = gradOutValue * (1 - m);
         // bool between(int value, int lowerBound, int upperBound)
         float v = 0;
         id_point = 0;
         for (int y = yi_t; y <= yi_b; ++y) {
           if (!between(y, 0, height-1)) continue;
           for (int x = xi_l; x <= xi_r; ++x) {
             if (!between(x, 0, width-1)) continue;
             int address = inputImages_strideBatch * b + inputImages_strideHeight * y + inputImages_strideWidth * x + t;
             v += weights[id_point] * inputImages_data[address];
             ++id_point;
           }
         }
         v /= weight_sum;
         c = canvas_data[gradOutputAddress + t];
         float gradMaskValue = gradOutValue * (v - c);

         id_point = 0;
         for (int y = yi_t; y <= yi_b; ++y) {
           if (!between(y, 0, height-1)) continue;
           for (int x = xi_l; x <= xi_r; ++x) {
             if (!between(x, 0, width-1)) continue;
             int address = inputImages_strideBatch * b + inputImages_strideHeight * y + inputImages_strideWidth * x;
             atomicAdd(&gradInputImages_data[address + t], weights[id_point] * gradOutValue_fg);
             atomicAdd(&gradMasks_data[address], weights[id_point] * gradMaskValue);
             ++id_point;
           }
         }
         gradCanvas_data[gradOutputAddress + t] = gradOutValue_bg;

         // compute gradient on grid coordinates
         id_point = 0;
         for (int y = yi_t; y <= yi_b; ++y) {
           if (!between(y, 0, height-1)) continue;
           for (int x = xi_l; x <= xi_r; ++x) {
             if (!between(x, 0, width-1)) continue;
             int address = inputImages_strideBatch * b + inputImages_strideHeight * y + inputImages_strideWidth * x;
             grad_yf += gradOutValue_fg * inputImages_data[address + t]
                        * weights[id_point]
                        * (bias_y[id_point] - bias_y_weighted / weight_sum)
                        / weight_sum;
             grad_xf += gradOutValue_fg * inputImages_data[address + t]
                        * weights[id_point]
                        * (bias_x[id_point] - bias_x_weighted / weight_sum)
                        / weight_sum;
             ++id_point;
           }
         }
       }
       gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = grad_yf * (inputImages_height-1) / 2;
       gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = grad_xf * (inputImages_width-1) / 2;
     }
   }
}

static int cunn_SubSamplerBHWD_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *masks = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *canvas = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");
  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 7, "torch.CudaTensor");
  THCudaTensor *gradMasks = (THCudaTensor *)luaT_checkudata(L, 8, "torch.CudaTensor");
  THCudaTensor *gradCanvas = (THCudaTensor *)luaT_checkudata(L, 9, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 10, "torch.CudaTensor");

   dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 threads(32,16);

   backwardSubSampling <false> <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (
                                                      THCudaTensor_data(state, inputImages),
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      THCudaTensor_data(state, gradInputImages),
                                                      THCudaTensor_stride(state, gradInputImages, 0),
                                                      THCudaTensor_stride(state, gradInputImages, 3),
                                                      THCudaTensor_stride(state, gradInputImages, 1),
                                                      THCudaTensor_stride(state, gradInputImages, 2),
                                                      THCudaTensor_data(state, grids),
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_data(state, gradGrids),
                                                      THCudaTensor_stride(state, gradGrids, 0),
                                                      THCudaTensor_stride(state, gradGrids, 3),
                                                      THCudaTensor_stride(state, gradGrids, 1),
                                                      THCudaTensor_stride(state, gradGrids, 2),
                                                      THCudaTensor_data(state, masks),
                                                      THCudaTensor_stride(state, masks, 0),
                                                      THCudaTensor_stride(state, masks, 3),
                                                      THCudaTensor_stride(state, masks, 1),
                                                      THCudaTensor_stride(state, masks, 2),
                                                      THCudaTensor_data(state, gradMasks),
                                                      THCudaTensor_stride(state, gradMasks, 0),
                                                      THCudaTensor_stride(state, gradMasks, 3),
                                                      THCudaTensor_stride(state, gradMasks, 1),
                                                      THCudaTensor_stride(state, gradMasks, 2),
                                                      THCudaTensor_data(state, canvas),
                                                      THCudaTensor_stride(state, canvas, 0),
                                                      THCudaTensor_stride(state, canvas, 3),
                                                      THCudaTensor_stride(state, canvas, 1),
                                                      THCudaTensor_stride(state, canvas, 2),
                                                      THCudaTensor_data(state, gradCanvas),
                                                      THCudaTensor_stride(state, gradCanvas, 0),
                                                      THCudaTensor_stride(state, gradCanvas, 3),
                                                      THCudaTensor_stride(state, gradCanvas, 1),
                                                      THCudaTensor_stride(state, gradCanvas, 2),
                                                      THCudaTensor_data(state, gradOutput),
                                                      THCudaTensor_stride(state, gradOutput, 0),
                                                      THCudaTensor_stride(state, gradOutput, 3),
                                                      THCudaTensor_stride(state, gradOutput, 1),
                                                      THCudaTensor_stride(state, gradOutput, 2),
                                                      THCudaTensor_size(state, inputImages, 3),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2),
                                                      THCudaTensor_size(state, gradOutput, 1),
                                                      THCudaTensor_size(state, gradOutput, 2));



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_BilinearSamplerBHWD_updateGradInputOnlyGrid(lua_State *L)
{
   THCState *state = getCutorchState(L);
   THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
   THCudaTensor *masks = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
   THCudaTensor *canvas = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
   THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");
   THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 7, "torch.CudaTensor");
   THCudaTensor *gradMasks = (THCudaTensor *)luaT_checkudata(L, 8, "torch.CudaTensor");
   THCudaTensor *gradCanvas = (THCudaTensor *)luaT_checkudata(L, 9, "torch.CudaTensor");
   THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 10, "torch.CudaTensor");

   dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 threads(32,16);

   backwardBilinearSampling <true> <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (
                                                      THCudaTensor_data(state, inputImages),
                                                      THCudaTensor_stride(state, inputImages, 0),
                                                      THCudaTensor_stride(state, inputImages, 3),
                                                      THCudaTensor_stride(state, inputImages, 1),
                                                      THCudaTensor_stride(state, inputImages, 2),
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      THCudaTensor_data(state, grids),
                                                      THCudaTensor_stride(state, grids, 0),
                                                      THCudaTensor_stride(state, grids, 3),
                                                      THCudaTensor_stride(state, grids, 1),
                                                      THCudaTensor_stride(state, grids, 2),
                                                      THCudaTensor_data(state, gradGrids),
                                                      THCudaTensor_stride(state, gradGrids, 0),
                                                      THCudaTensor_stride(state, gradGrids, 3),
                                                      THCudaTensor_stride(state, gradGrids, 1),
                                                      THCudaTensor_stride(state, gradGrids, 2),
                                                      THCudaTensor_data(state, masks),
                                                      THCudaTensor_stride(state, masks, 0),
                                                      THCudaTensor_stride(state, masks, 3),
                                                      THCudaTensor_stride(state, masks, 1),
                                                      THCudaTensor_stride(state, masks, 2),
                                                      THCudaTensor_data(state, gradMasks),
                                                      THCudaTensor_stride(state, gradMasks, 0),
                                                      THCudaTensor_stride(state, gradMasks, 3),
                                                      THCudaTensor_stride(state, gradMasks, 1),
                                                      THCudaTensor_stride(state, gradMasks, 2),
                                                      THCudaTensor_data(state, canvas),
                                                      THCudaTensor_stride(state, canvas, 0),
                                                      THCudaTensor_stride(state, canvas, 3),
                                                      THCudaTensor_stride(state, canvas, 1),
                                                      THCudaTensor_stride(state, canvas, 2),
                                                      THCudaTensor_data(state, gradCanvas),
                                                      THCudaTensor_stride(state, gradCanvas, 0),
                                                      THCudaTensor_stride(state, gradCanvas, 3),
                                                      THCudaTensor_stride(state, gradCanvas, 1),
                                                      THCudaTensor_stride(state, gradCanvas, 2),
                                                      THCudaTensor_data(state, gradOutput),
                                                      THCudaTensor_stride(state, gradOutput, 0),
                                                      THCudaTensor_stride(state, gradOutput, 3),
                                                      THCudaTensor_stride(state, gradOutput, 1),
                                                      THCudaTensor_stride(state, gradOutput, 2),
                                                      THCudaTensor_size(state, inputImages, 3),
                                                      THCudaTensor_size(state, inputImages, 1),
                                                      THCudaTensor_size(state, inputImages, 2),
                                                      THCudaTensor_size(state, gradOutput, 2));



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}



static const struct luaL_Reg cunn_BilinearSamplerBHWD__ [] = {
  {"BilinearSamplerBHWD_updateOutput", cunn_BilinearSamplerBHWD_updateOutput},
  {"BilinearSamplerBHWD_updateGradInput", cunn_BilinearSamplerBHWD_updateGradInput},
  {"SubSamplerBHWD_updateOutput", cunn_SubSamplerBHWD_updateOutput},
  {"SubSamplerBHWD_updateGradInput", cunn_SubSamplerBHWD_updateGradInput},
  {"BilinearSamplerBHWD_updateGradInputOnlyGrid", cunn_BilinearSamplerBHWD_updateGradInputOnlyGrid},
  {NULL, NULL}
};

static void cunn_BilinearSamplerBHWD_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_BilinearSamplerBHWD__, "nn");
  lua_pop(L,1);
}
