#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include "libs/bitmap.h"

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5
// If you apply another kernel, remember not only to exchange
// the kernel but also the kernelFactor and the correct dimension.

int const sobelYKernel[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYKernelFactor = (float) 1.0;

int const sobelXKernel[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXKernelFactor = (float) 1.0;


int const laplacian1Kernel[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};

float const laplacian1KernelFactor = (float) 1.0;

int const laplacian2Kernel[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2KernelFactor = (float) 1.0;

int const laplacian3Kernel[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3KernelFactor = (float) 1.0;


//Bonus Kernel:

int const gaussianKernel[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };

float const gaussianKernelFactor = (float) 1.0 / 256.0;


// Helper function to swap bmpImageChannel pointers
void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two) {
  bmpImageChannel *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char **out, unsigned char **in,
                 unsigned int width, unsigned int height,
                 unsigned int startX, unsigned int startY,
                 int *kernel, unsigned int kernelDim, float kernelFactor) {
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = startY; y < height; y++) {
    for (unsigned int x = startX; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
        }
      }
      aggregate *= kernelFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
      }
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

int main(int argc, char **argv) {
    // Set up MPI
    MPI_Init(NULL, NULL);

    // Find the rank of this process and the number of processes
    int rank;
    int numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    printf("MPI is set up. My rank: %d, number of processes: %d\n", rank, numProcesses);


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
        goto graceful_exit;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    goto error_exit;
  }
  input = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */









  // Full image, only exists on main rank
  bmpImage *image;
  // Full grayscale image, only on main rank
  bmpImageChannel *imageChannel;

  // Number of rows in a chunk *excluding* borders
  int chunkHeight;
  // Number of rows in the main proc's chunk (has any non-divisible rows as well) *excluding* borders
  int chunkHeightMaster;

  // The local image chunk each rank will work on
  bmpImageChannel* localChunk;
  int topBorder = 0;
  int bottomBorder = 0;

  clock_t time_start;

  // Main process reads the image from disk
  if (rank == 0) {
    /*
        Create the BMP image and load it from disk.
    */
    image = newBmpImage(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
    }

    if (loadBmpImage(image, input) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        freeBmpImage(image);
        goto error_exit;
    }


    // Create a single color channel image. It is easier to work just with one color
    imageChannel = newBmpImageChannel(image->width, image->height);
    if (imageChannel == NULL) {
        fprintf(stderr, "Could not allocate new image channel!\n");
        freeBmpImage(image);
        goto error_exit;
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
      goto error_exit;
    }

    printf("Image width: %d, height: %d, data: %d\n", imageChannel->width, imageChannel->height, imageChannel->width*imageChannel->height);



    time_start = clock();

    chunkHeight = imageChannel->height / numProcesses;
    chunkHeightMaster = chunkHeight + (imageChannel->height % numProcesses);

    int dataOffset = (chunkHeightMaster - 0) * imageChannel->width;
    int chunkData = imageChannel->width * chunkHeight;

    for (int i = 1; i < numProcesses; ++i) {
      int bottom = 1;
      int top = i != (numProcesses - 1);
      int h = chunkHeight + top + bottom;
      int meta[] = { imageChannel->width, h, top, bottom };

      // Send the chunk dimensions
      MPI_Send(&meta, 4, MPI_INT, i, 0, MPI_COMM_WORLD);
      // Send the actual chunk data
      MPI_Send(imageChannel->rawdata + dataOffset, imageChannel->width * h, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
      dataOffset += chunkData;
    }
    printf("Main sent %d data\n", dataOffset);

    localChunk = newBmpImageChannel(imageChannel->width, chunkHeightMaster + 1);
    memcpy(localChunk->rawdata, imageChannel->rawdata, imageChannel->width * (chunkHeightMaster + 1));
    printf("Main must process %d data\n\n", localChunk->height*localChunk->width);
    topBorder = 1;
  }
  else {
    // The workers
    int meta[4];
    MPI_Recv(&meta, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    localChunk = newBmpImageChannel(meta[0], meta[1]);
    topBorder = meta[2];
    bottomBorder = meta[3];

    MPI_Recv(localChunk->rawdata, meta[0] * meta[1], MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Rank %d got %d data and meta (%d %d %d %d)\n", rank, meta[0] * meta[1], meta[0], meta[1], meta[2], meta[3]);
  }



  printf("Rank %d topborder: %d, bottomborder: %d\n", rank, topBorder, bottomBorder);

  //Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])
  bmpImageChannel *localChunkOut = newBmpImageChannel(localChunk->width, localChunk->height);
  applyKernel(localChunkOut->data,
              localChunk->data,
              localChunk->width,
              localChunk->height - topBorder,
              0, bottomBorder,
              (int *)laplacian1Kernel, 3, laplacian1KernelFactor
              // (int *)laplacian2Kernel, 3, laplacian2KernelFactor
              // (int *)laplacian3Kernel, 3, laplacian3KernelFactor
              // (int *)gaussianKernel, 5, gaussianKernelFactor
              );
  swapImageChannel(&localChunkOut, &localChunk);
  freeBmpImageChannel(localChunkOut);



  // Main process saves the image to disk
  if (rank == 0) {
    int dataOffset = imageChannel->width * chunkHeightMaster;
    int chunkData = imageChannel->width * chunkHeight;
    for (int i = 1; i < numProcesses; ++i) {
      // Receive the processed chunk data
      MPI_Recv(imageChannel->rawdata + dataOffset, imageChannel->width * chunkHeight, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      dataOffset += chunkData;
    }
    printf("Main got all data\n");
    memcpy(imageChannel->rawdata, localChunk->rawdata, imageChannel->width * chunkHeightMaster);

    double elapsed = (double) (clock() - time_start) / CLOCKS_PER_SEC;
    printf("TIME USED: %f\n\n", elapsed);



    // Map our single color image back to a normal BMP image with 3 color channels
    // mapEqual puts the color value on all three channels the same way
    // other mapping functions are mapRed, mapGreen, mapBlue
    if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
      fprintf(stderr, "Could not map image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(imageChannel);
      goto error_exit;
    }
    freeBmpImageChannel(imageChannel);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };
  }
  else {
    // Send the processed chunk data
    MPI_Send(localChunk->rawdata + (topBorder * localChunk->width), localChunk->width * (localChunk->height - topBorder - bottomBorder), MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
    printf("Rank %d returned the modified data\n", rank);
  }

graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);


  free(localChunk);

  MPI_Finalize();
  return ret;
};