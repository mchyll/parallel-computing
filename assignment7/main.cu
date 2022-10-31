/*
PARALLEL COMPUTING - ASSIGNMENT 7
Magnus Conrad Hyll
*/

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <sys/time.h>
extern "C" {
    #include "libs/bitmap.h"
}
#include <cuda_runtime_api.h>

#define ERROR_EXIT -1

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5
// If you apply another filter, remember not only to exchange
// the filter but also the filterFactor and the correct dimension.

int const sobelYFilter[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYFilterFactor = (float) 1.0;

int const sobelXFilter[] = {1, -0, -1,
                            2,  0, -2,
                            1,  0, -1 , 0};
float const sobelXFilterFactor = (float) 1.0;


int const laplacian1Filter[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};

float const laplacian1FilterFactor = (float) 1.0;

int const laplacian2Filter[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2FilterFactor = (float) 1.0;

int const laplacian3Filter[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3FilterFactor = (float) 1.0;


//Bonus Filter:

int const gaussianFilter[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };

float const gaussianFilterFactor = (float) 1.0 / 256.0;


// Apply convolutional filter on image data
void applyFilter(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, const int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy][xx] * filter[nky * filterDim + nkx];
        }
      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
      }
    }
  }
}

// Apply convolutional filter on image data
__global__ void cudaApplyFilter(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  int x = threadIdx.x + blockIdx.x * blockDim.x;  // x coordinate of pixel
  int y = threadIdx.y + blockIdx.y * blockDim.y;  // y coordinate of pixel

  // Check if point is within image
  if (x < width && y < height) {
    int aggregate = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
      int nky = filterDim - 1 - ky;
      for (unsigned int kx = 0; kx < filterDim; kx++) {
        int nkx = filterDim - 1 - kx;

        int yy = y + (ky - filterCenter);
        int xx = x + (kx - filterCenter);
        if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
          aggregate += in[yy * width + xx] * filter[nky * filterDim + nkx];
      }
    }
    aggregate *= filterFactor;
    if (aggregate > 0) {
      out[y * width + x] = (aggregate > 255) ? 255 : aggregate;
    } else {
      out[y * width + x] = 0;
    }
  }
}

// Apply convolutional filter on image data
__global__ void cudaApplyFilterSharedMem(unsigned char *out, unsigned char *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);
  int x = threadIdx.x + blockIdx.x * blockDim.x;  // x coordinate of pixel
  int y = threadIdx.y + blockIdx.y * blockDim.y;  // y coordinate of pixel

  // The thread block cache containing part of the image
  __shared__ unsigned char blockCache[BLOCK_WIDTH * BLOCK_HEIGHT];
  // The filter, cached to shared memory. The size of this array is determined at kernel launch
  extern __shared__ int filterCache[];

  // Copy filter to shared memory
  if (threadIdx.x < filterDim && threadIdx.y < filterDim)
    filterCache[threadIdx.y * filterDim + threadIdx.x] = filter[threadIdx.y * filterDim + threadIdx.x];

  // Copy image block to shared memory
  if (x < width && y < height)
    blockCache[threadIdx.y * blockDim.x + threadIdx.x] = in[y * width + x];

  // Wait until all threads have finished copying to shared mem
  __syncthreads();

  // Check if point is within image
  if (x < width && y < height) {
    int aggregate = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
      int nky = filterDim - 1 - ky;
      for (unsigned int kx = 0; kx < filterDim; kx++) {
        int nkx = filterDim - 1 - kx;

        // Check first if current pixel is within block cache
        int block_x = threadIdx.x + (kx - filterCenter);
        int block_y = threadIdx.y + (ky - filterCenter);
        if (block_x >= 0 && block_x < BLOCK_WIDTH && block_y >= 0 && block_y < BLOCK_HEIGHT) {
          aggregate += blockCache[block_y * BLOCK_WIDTH + block_x] * filterCache[nky * filterDim + nkx];
        }
        // If not, do the usual and read from global memory
        else {
          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy * width + xx] * filterCache[nky * filterDim + nkx];
        }
      }
    }
    aggregate *= filterFactor;
    if (aggregate > 0) {
      out[y * width + x] = (aggregate > 255) ? 255 : aggregate;
    } else {
      out[y * width + x] = 0;
    }
  } 
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

void runSerial(bmpImageChannel* imageChannel, int iterations, const int* filter, int filterDim, float filterFactor) {
  //Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])
  bmpImageChannel *processImageChannel = newBmpImageChannel(imageChannel->width, imageChannel->height);
  for (unsigned int i = 0; i < iterations; i ++) {
    applyFilter(processImageChannel->data,
                imageChannel->data,
                imageChannel->width,
                imageChannel->height,
                filter, filterDim, filterFactor);

    // Swap the data pointers
    unsigned char ** tmp = processImageChannel->data;
    processImageChannel->data = imageChannel->data;
    imageChannel->data = tmp;
    unsigned char * tmp_raw = processImageChannel->rawdata;
    processImageChannel->rawdata = imageChannel->rawdata;
    imageChannel->rawdata = tmp_raw;
  }
  freeBmpImageChannel(processImageChannel);
}

void runCuda(bmpImageChannel* imageChannel, int iterations, const int* filter, int filterDim, float filterFactor) {
  // Allocate memory for input image, output image and filter on GPU
  unsigned char* inImage;
  cudaErrorCheck(cudaMalloc((void**) &inImage, imageChannel->width * imageChannel->height * sizeof(int)));
  unsigned char* outImage;
  cudaErrorCheck(cudaMalloc((void**) &outImage, imageChannel->width * imageChannel->height * sizeof(int)));
  int* deviceFilter;
  cudaErrorCheck(cudaMalloc((void**) &deviceFilter, filterDim * filterDim * sizeof(int)));

  // Copy data for original image and filter to GPU
  cudaErrorCheck(cudaMemcpy(inImage, imageChannel->rawdata, imageChannel->width * imageChannel->height * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(deviceFilter, filter, filterDim * filterDim * sizeof(int), cudaMemcpyHostToDevice));

  // Define dimensions for block-grid and thread-blocks
  dim3 gridDim(imageChannel->width / BLOCK_WIDTH + 1, imageChannel->height / BLOCK_HEIGHT + 1); // Grid consists of blocks
  dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT); // Block consists of threads

  // Here we do the actual computation!
  for (unsigned int i = 0; i < iterations; i++) {
    // Thrid launch argument is the size of the array in shared memory containing the filter
    cudaApplyFilterSharedMem<<<gridDim, blockDim, (filterDim*filterDim)>>>(outImage, inImage, imageChannel->width, imageChannel->height, deviceFilter, filterDim, filterFactor);
    // cudaApplyFilter<<<gridDim, blockDim>>>(outImage, inImage, imageChannel->width, imageChannel->height, deviceFilter, filterDim, filterFactor);

    // Swap the data pointers
    unsigned char* tmp = inImage;
    inImage = outImage;
    outImage = tmp;
  }

  // Copy resulting image back to main memory
  cudaErrorCheck(cudaMemcpy(imageChannel->rawdata, inImage, imageChannel->width * imageChannel->height * sizeof(int), cudaMemcpyDeviceToHost));

  // Free the GPU-allocated memory
  cudaErrorCheck(cudaFree(inImage));
  cudaErrorCheck(cudaFree(outImage));
  cudaErrorCheck(cudaFree(deviceFilter));
}

double walltime() {
	static struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + 1e-6 * t.tv_usec;
}

int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hi:";
  {
    char *endptr;
    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        return 0;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          return ERROR_EXIT;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    return ERROR_EXIT;
  }
  input = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */

  /*
    Create the BMP image and load it from disk.
   */
  bmpImage *image = newBmpImage(0,0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    return ERROR_EXIT;
  }


  // Create a single color channel image. It is easier to work just with one color
  bmpImageChannel *imageChannel = newBmpImageChannel(image->width, image->height);
  if (imageChannel == NULL) {
    fprintf(stderr, "Could not allocate new image channel!\n");
    freeBmpImage(image);
    return ERROR_EXIT;
  }

  // Extract from the loaded image an average over all colors - nothing else than
  // a black and white representation
  // extractImageChannel and mapImageChannel need the images to be in the exact
  // same dimensions!
  // Other prepared extraction functions are extractRed, extractGreen, extractBlue
  if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
    fprintf(stderr, "Could not extract image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }

  const int* filter = laplacian1Filter;
  int filterDim = 3;
  float filterFactor = laplacian1FilterFactor;

  double tStart = walltime();

  // runSerial(imageChannel, iterations, filter, filterDim, filterFactor);
  runCuda(imageChannel, iterations, filter, filterDim, filterFactor);

  printf("Time: %.5f sec\n", walltime() - tStart);

  // Map our single color image back to a normal BMP image with 3 color channels
  // mapEqual puts the color value on all three channels the same way
  // other mapping functions are mapRed, mapGreen, mapBlue
  if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
    fprintf(stderr, "Could not map image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }
  freeBmpImageChannel(imageChannel);

  // Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    return ERROR_EXIT;
  };

  ret = 0;
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
};
