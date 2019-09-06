#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before fullOriginalImage
#define YSIZE 2048

uchar* modifyImage(uchar* fullOriginalImage, int xSize, int ySize);

int main() {
	// printf("Test reading image\n");
	// uchar* lol = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	// readbmp("before.bmp", lol);
	// free(lol);
	// printf("DONE Test reading image\n");

    // Set up MPI
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	printf("MPI is set up. My rank: %d, num procs: %d\n", rank, numProcesses);

    if (YSIZE % numProcesses != 0) {
        printf("Can't divide image in equal parts for each process. Try a number of processes which divides the number of rows in the image.\n");
        MPI_Finalize();
        return 1;
    }

    int rowsPerProcess = YSIZE / numProcesses;
    int bytesPerProcess = XSIZE * YSIZE * 3 / numProcesses;

	printf("Rows per proc: %d, bytes per proc: %d\n", rowsPerProcess, bytesPerProcess);

    uchar* fullOriginalImage = NULL;
    uchar* localOriginalImage = NULL;

    if (rank == 0) {
		printf("Rank 0 reading image\n");
        *fullOriginalImage = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
        readbmp("before.bmp", fullOriginalImage);
		printf("Done reading image\n");
    }

	printf("Scattering in rank %d\n", rank);
    MPI_Scatter(fullOriginalImage, bytesPerProcess, MPI_UNSIGNED_CHAR, localOriginalImage, bytesPerProcess, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    free(fullOriginalImage);

	printf("Modifying local image\n");
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
