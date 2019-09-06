#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before fullOriginalImage
#define YSIZE 2048

uchar* modifyImage(uchar* fullOriginalImage, int xSize, int ySize);

int main() {
    // Set up MPI
    MPI_Init(NULL, NULL);

    // Find the rank of this process and the number of processes
    int rank;
    int numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    printf("MPI is set up. My rank: %d, number of processes: %d\n", rank, numProcesses);

    // Make sure we can divide the image evenly
    if (YSIZE % numProcesses != 0) {
        printf("Can't divide image in equal parts for each process. Try a number of processes which divides the number of rows in the image.\n");
        MPI_Finalize();
        return 1;
    }

    // Find number of rows and bytes each worker should process
    int rowsPerProcess = YSIZE / numProcesses;
    int bytesPerProcess = XSIZE * YSIZE * 3 / numProcesses;

    uchar* fullOriginalImage;
    uchar* localOriginalImage = calloc(XSIZE * rowsPerProcess * 3, 1);

    // Main process reads the original image
    if (rank == 0) {
        printf("Rank %d reading image\n", rank);
        fullOriginalImage = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
        readbmp("before.bmp", fullOriginalImage);
    }

    // Main process scatters image data across processes,
    // processes receives their part of the image into the localOriginalImage buffer
    printf("Scattering in rank %d\n", rank);
    MPI_Scatter(fullOriginalImage, bytesPerProcess, MPI_UNSIGNED_CHAR, localOriginalImage, bytesPerProcess, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Each process modifies its part of the image
    printf("Modifying local image in rank %d\n", rank);
    uchar* localModifiedImage = modifyImage(localOriginalImage, XSIZE, rowsPerProcess);
    free(localOriginalImage);

    uchar* fullModifiedImage;
    if (rank == 0) {
        // Only main process should allocate memory for the full modified image
        fullModifiedImage = calloc(4 * XSIZE * YSIZE * 3, 1);
    }

    // Processes sends their part of the image from the localModifiedImage buffer,
    // main process gathers all image data from the processes
    printf("Gathering full image in rank %d\n", rank);
    MPI_Gather(localModifiedImage, bytesPerProcess * 4, MPI_UNSIGNED_CHAR, fullModifiedImage, bytesPerProcess * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Main process saves the full modified image
    if (rank == 0) {
        printf("Saving full image in rank %d\n", rank);
        savebmp("after.bmp", fullModifiedImage, XSIZE * 2, YSIZE * 2);

        // Free the memory allocated for the full original and modified images
        free(fullOriginalImage);
        free(fullModifiedImage);
    }

    free(localModifiedImage);
    MPI_Finalize();

    return 0;
}

/**
 * Modifies an image by doubling the resolution and swapping color channels
 */
uchar* modifyImage(uchar* image, int xSize, int ySize) {
    uchar* modifiedImage = calloc(4 * xSize * ySize * 3, 1); // Modified image is double the size

    // Loop through all pixels in the up-scaled image
    for (int x = 0; x < xSize * 2; ++x) {
        for (int y = 0; y < ySize * 2; ++y) {
            // Index of the first color component of the current pixel in the original image array
            int indexOriginal = ((int)(y / 2) * xSize + (int)(x / 2)) * 3;

            // Index of the first color component of the current pixel in the up-scaled image array
            int indexModified = (y * xSize * 2 + x) * 3;

            // Simple manipulation; swaps RGB color components around
            modifiedImage[indexModified] = image[indexOriginal + 1];
            modifiedImage[indexModified + 1] = image[indexOriginal + 2];
            modifiedImage[indexModified + 2] = image[indexOriginal];
        }
    }

    return modifiedImage;
}
