#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

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


int const _laplacian1Kernel[] = {  0,  1,  2,
                                 3,  4,  5,
                                 6,  7,  8};
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
  for (unsigned int y = startY; y < height; ++y) {
    for (unsigned int x = startX; x < width; ++x) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < kernelDim; ++ky) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; ++kx) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)  // TODO: Her tror kernel at bildet stopper en rad for tidlig i toppen, dersom jeg sier at height=chunk.height-topBorder
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

typedef struct {
  // Width of image chunk
  int width;
  // Height of image chunk, excluding borders
  int height;
  // Number of top border rows in the image chunk data
  int topBorder;
  // Number of bottom border rows in the image chunk data
  int bottomBorder;
  // Starting row number (y-value) in the full image
  int origRow;
} ChunkMeta;

void convolute(bmpImageChannel** localChunk, const ChunkMeta* meta, int iterations);
void borderExchange(bmpImageChannel* localChunk, const ChunkMeta* meta);

int rank;
int numProcesses;

int main(int argc, char **argv) {
  // Set up MPI
  MPI_Init(NULL, NULL);

  // Find the rank of this process and the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

  printf("MPI is set up. My rank: %d, number of processes: %d, PID: %d\n", rank, numProcesses, getpid());
#ifdef DEBUG
  sleep(30);
#endif

  ChunkMeta chunkMetas[numProcesses];

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




  // The local image chunk each rank will work on
  bmpImageChannel* localChunk;

  // Main process reads the image from disk
  if (rank == 0) {
    /*
        Create the BMP image and load it from disk.
    */
    // Full image, only exists on main rank
    bmpImage* image = newBmpImage(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
    }

    if (loadBmpImage(image, input) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        freeBmpImage(image);
        goto error_exit;
    }


    // Create a single color channel image. It is easier to work just with one color
    // Full grayscale image, only on main rank
    bmpImageChannel* imageChannel = newBmpImageChannel(image->width, image->height);
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

    struct timespec time_start;
    clock_gettime(CLOCK_REALTIME, &time_start);
    double t_start = time_start.tv_sec + (double)time_start.tv_nsec / 1e9;

    // Number of rows in a chunk *excluding* borders
    int chunkHeight = imageChannel->height / numProcesses;
    // Number of rows in the main proc's chunk (has any non-divisible rows as well) *excluding* borders
    int chunkHeightMaster = chunkHeight + (imageChannel->height % numProcesses);

    int borderSize = 1;

    // The offset of the whole image data to start sending from
    int dataOffset = imageChannel->width * (chunkHeightMaster - borderSize);
    int chunkBytes = imageChannel->width * chunkHeight;

    for (int i = 1; i < numProcesses; ++i) {
      int bottom = borderSize;
      int top = i != (numProcesses - 1) ? borderSize : 0;
      int dataHeight = chunkHeight + top + bottom;

      // Chunk metadata: {chunk width, chunk height, num rows top border, num rows bottom border}
      chunkMetas[i] = (ChunkMeta) {
        imageChannel->width, chunkHeight,
        top, bottom
      };

      // Send the chunk dimensions
      MPI_Send(&chunkMetas[i], sizeof(ChunkMeta), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
      // Send the actual chunk data
      MPI_Send(imageChannel->rawdata + dataOffset, imageChannel->width * (chunkHeight + top + bottom), MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
      dataOffset += chunkBytes;
    }
    printf("Main sent %d data\n", dataOffset);

    // "Send" data to main proc as well
    ChunkMeta meta = {imageChannel->width, chunkHeightMaster, .topBorder = 1, .bottomBorder = 0};
    printf("Rank %d got meta (%d %d %d %d)\n", rank, meta.width, meta.height, meta.topBorder, meta.bottomBorder);
    int dataHeight = chunkHeightMaster + meta.topBorder + meta.bottomBorder;
    localChunk = newBmpImageChannel(imageChannel->width, dataHeight);
    memcpy(localChunk->rawdata, imageChannel->rawdata, imageChannel->width * dataHeight);
    printf("Main must process %d data\n", localChunk->height*localChunk->width);

    printf("Rank %d: local chunk is (%d x %d)\n\n", rank, localChunk->width, localChunk->height);
    convolute(&localChunk, &meta, iterations);
    // printf("Rank %d: local chunk is (%d x %d)\n", rank, localChunk->width, localChunk->height);

    dataOffset = imageChannel->width * (chunkHeightMaster);
    for (int i = 1; i < numProcesses; ++i) {
      // Receive the processed chunk data
      MPI_Recv(imageChannel->rawdata + dataOffset, imageChannel->width * chunkHeight, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      dataOffset += chunkBytes;
    }
    printf("Main got all data\n");
    memcpy(imageChannel->rawdata, localChunk->rawdata, imageChannel->width * chunkHeightMaster);

    struct timespec time_end;
    clock_gettime(CLOCK_REALTIME, &time_end);
    double t_end = time_end.tv_sec + (double)time_end.tv_nsec / 1e9;
    double elapsed = t_end - t_start;
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

  // Procedure for all non-main ranks
  else {
    ChunkMeta meta;
    MPI_Recv(&meta, sizeof(ChunkMeta), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Rank %d got meta (%d %d %d %d)\n", rank, meta.width, meta.height, meta.topBorder, meta.bottomBorder);

    int dataHeight = meta.height + meta.topBorder + meta.bottomBorder;
    localChunk = newBmpImageChannel(meta.width, dataHeight);

    MPI_Recv(localChunk->rawdata, meta.width * dataHeight, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Rank %d got %d data\n", rank, meta.width*dataHeight);

    printf("Rank %d: local chunk is (%d x %d)\n", rank, localChunk->width, localChunk->height);
    convolute(&localChunk, &meta, iterations);

    printf("Rank %d will return %d data\n", rank, meta.width * meta.height);
    // printf("Rank %d: local chunk is (%d x %d)\n", rank, localChunk->width, localChunk->height);
    // Send the processed chunk data
    MPI_Send(localChunk->rawdata + (meta.bottomBorder * localChunk->width), meta.width * meta.height, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
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

void convolute(bmpImageChannel** localChunkPtr, const ChunkMeta* meta, int iterations) {
  //Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])

  // Helper pointer so we don't have to double-dereference localChunkPtr all the time
  bmpImageChannel* localChunk = *localChunkPtr;
  bmpImageChannel* localChunkOut = newBmpImageChannel(localChunk->width, localChunk->height);
  for (int i = 0; i < iterations; ++i) {
    applyKernel(localChunkOut->data,
                localChunk->data,
                localChunk->width,
                localChunk->height,
                0, 0,
                // localChunk->height - meta->topBorder,
                // meta->height,
                // 0, meta->bottomBorder,
                (int *)laplacian1Kernel, 3, laplacian1KernelFactor
                // (int *)laplacian2Kernel, 3, laplacian2KernelFactor
                // (int *)laplacian3Kernel, 3, laplacian3KernelFactor
                // (int *)gaussianKernel, 5, gaussianKernelFactor
                );
    swapImageChannel(&localChunkOut, localChunkPtr);
    // bmpImageChannel* tmp = *localChunkPtr;
    // *localChunkPtr = localChunkOut;
    // localChunkOut = tmp;
    localChunk = *localChunkPtr;

    borderExchange(localChunk, meta);
  }

  freeBmpImageChannel(localChunkOut);
}

void borderExchange(bmpImageChannel* localChunk, const ChunkMeta* meta) {
  // EVEN RANKS start by sending their top rows up
  // sleep(rank);
  if (rank % 2 == 0) {
    // 1. SEND TOP EDGE INTO BOTTOM BORDER OF RANK ABOVE
    // The topmost rank can't send up
    if (rank != numProcesses - 1) {
      // printf("1: Rank %d sending top edge up\n", rank);fflush(stdout);
      MPI_Send(
        localChunk->rawdata + meta->width * (meta->bottomBorder + meta->height - meta->topBorder),
        meta->width * meta->topBorder, MPI_UNSIGNED_CHAR, rank + 1, 11, MPI_COMM_WORLD);
    }

    // sleep(rank + numProcesses + 1);

    // 2. RECEIVE BOTTOM BORDER FROM TOP EDGE OF RANK BELOW
    // Rank 0 can't receive from rank below
    if (rank != 0) {
      // printf("2: Rank %d receiving bottom border from below\n", rank);fflush(stdout);
      MPI_Recv(localChunk->rawdata, meta->width * meta->bottomBorder, MPI_UNSIGNED_CHAR, rank - 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // sleep(rank + numProcesses + 1);

    // 3. SEND BOTTOM EDGE INTO TOP BORDER OF RANK BELOW
    // Rank 0 can't send down
    if (rank != 0) {
      // printf("3: Rank %d sending bottom edge down\n", rank);fflush(stdout);
      MPI_Send(localChunk->rawdata + meta->width * meta->bottomBorder, meta->width * meta->bottomBorder, MPI_UNSIGNED_CHAR, rank - 1, 13, MPI_COMM_WORLD);
    }

    // sleep(rank + numProcesses + 1);

    // 4. RECEIVE TOP BORDER FROM BOTTOM EDGE OF RANK ABOVE
    // The topmost rank can't receive from rank above
    if (rank != numProcesses - 1) {
      // printf("4: Rank %d receiving top border from above\n", rank);fflush(stdout);
      MPI_Recv(
        localChunk->rawdata + meta->width * (meta->bottomBorder + meta->height),
        meta->width * meta->topBorder, MPI_UNSIGNED_CHAR, rank + 1, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // ODD RANKS start by receiving bottom border
  else {
    // 1. RECEIVE BOTTOM BORDER FROM TOP EDGE OF RANK BELOW
    // Rank 0 can't receive from rank below
    if (rank != 0) {
      // printf("1: Rank %d receiving bottom border from below\n", rank);fflush(stdout);
      MPI_Recv(localChunk->rawdata, meta->width * meta->bottomBorder, MPI_UNSIGNED_CHAR, rank - 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // sleep(rank + numProcesses + 1);

    // 2. SEND TOP EDGE INTO BOTTOM BORDER OF RANK ABOVE
    // The topmost rank can't send up
    if (rank != numProcesses - 1) {
      // printf("2: Rank %d sending top edge up\n", rank);fflush(stdout);
      MPI_Send(
        localChunk->rawdata + meta->width * (meta->bottomBorder + meta->height - meta->topBorder),
        meta->width * meta->topBorder, MPI_UNSIGNED_CHAR, rank + 1, 12, MPI_COMM_WORLD);
    }

    // sleep(rank + numProcesses + 1);

    // 3. RECEIVE TOP BORDER FROM BOTTOM EDGE OF RANK ABOVE
    // The topmost rank can't receive from rank above
    if (rank != (numProcesses - 1)) {
      // printf("3: Rank %d receiving top border from above\n", rank);fflush(stdout);
      MPI_Recv(
        localChunk->rawdata + meta->width * (meta->bottomBorder + meta->height),
        meta->width * meta->topBorder, MPI_UNSIGNED_CHAR, rank + 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // sleep(rank + numProcesses + 1);

    // 4. SEND BOTTOM EDGE INTO TOP BORDER OF RANK BELOW
    // Rank 0 can't send down
    if (rank != 0) {
      // printf("4: Rank %d sending bottom edge down\n", rank);fflush(stdout);
      MPI_Send(
        localChunk->rawdata + meta->width * meta->bottomBorder,
        meta->width * meta->bottomBorder, MPI_UNSIGNED_CHAR, rank - 1, 14, MPI_COMM_WORLD);
    }
  }
}