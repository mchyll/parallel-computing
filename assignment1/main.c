#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before fullOriginalImage
#define YSIZE 2048

uchar* modifyImage(uchar* fullOriginalImage, int xSize, int ySize);

int main() {
    int rank;
    int numProcesses;

    // Set up MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    if (numProcesses % YSIZE != 0) {
        printf("Can't divide image in equal parts for each process. Try a number of processes which divides the number of rows in the image.\n");
        MPI_Finalize();
        return 1;
    }

    int rowsPerProcess = YSIZE / numProcesses;
    int bytesPerProcess = XSIZE * YSIZE * 3 / numProcesses;

    uchar* fullOriginalImage;
    uchar* localOriginalImage;

    if (rank == 0) {
        *fullOriginalImage = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
        readbmp("before.bmp", fullOriginalImage);
    }

    MPI_Scatter(fullOriginalImage, bytesPerProcess, MPI_UNSIGNED_CHAR, localOriginalImage, bytesPerProcess, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    free(fullOriginalImage);

    uchar* localModifiedImage = modifyImage(localOriginalImage, XSIZE, rowsPerProcess);
    free(localOriginalImage);

    if (rank == 0) {
        uchar* fullModifiedImage = calloc(4 * XSIZE * YSIZE * 3, 1);
        MPI_Gather(localModifiedImage, bytesPerProcess * 4, MPI_UNSIGNED_CHAR, fullModifiedImage, bytesPerProcess * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        savebmp("after.bmp", fullModifiedImage, XSIZE * 2, YSIZE * 2);
    }
    free(localModifiedImage);

    return 0;
}

uchar* modifyImage(uchar* fullOriginalImage, int xSize, int ySize) {
    uchar* modifiedImage = calloc(4 * xSize * ySize * 3, 1); // Modified fullOriginalImage is double the size

    // Loop through all pixels in the up-scaled fullOriginalImage
    for (int x = 0; x < xSize * 2; ++x) {
        for (int y = 0; y < ySize * 2; ++y) {
            // Index of the first color component of the current pixel in the original fullOriginalImage array
            int indexOriginal = ((int)(y / 2) * xSize + (int)(x / 2)) * 3;

            // Index of the first color component of the current pixel in the up-scaled fullOriginalImage array
            int indexModified = (y * xSize * 2 + x) * 3;

            // Simple manipulation; swaps RGB color components around
            modifiedImage[indexModified] = fullOriginalImage[indexOriginal + 1];
            modifiedImage[indexModified + 1] = fullOriginalImage[indexOriginal + 2];
            modifiedImage[indexModified + 2] = fullOriginalImage[indexOriginal];
        }
    }

    return modifiedImage;
}
