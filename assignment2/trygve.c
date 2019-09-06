#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

int main() {
    int rank, num_proc, data_per_proc, lines_per_proc;

    //Initializing MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    //Checking if number of lines is divisible by number of processes
    if (fabs((double) YSIZE / num_proc - floor(((double) YSIZE / num_proc))) > 0.0000001) {
        printf("Number of lines not divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    //Calculating lines and amount of data to each process
    data_per_proc = (XSIZE * YSIZE * 3 / num_proc);
    lines_per_proc = (YSIZE / num_proc);

    //allocating local part of image for each process
    uchar *local_image = calloc(XSIZE * lines_per_proc * 3, 1);

    if (rank == 0) {
        //Allocates memory for image and loads it from disk
        uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
        readbmp("before.bmp", image);

        //Splits up the image between the processes
        MPI_Scatter(image, data_per_proc, MPI_UNSIGNED_CHAR, local_image, data_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        free(image);
    } else {
        //Recives data from process 0
        MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, local_image, data_per_proc, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    //Do processing start
    blueToGreen(local_image, lines_per_proc, XSIZE); //swaps blue and green colors

    uchar *local_double_image = calloc(XSIZE * 2 * lines_per_proc * 2 * 3, 1);
    resolutionDoubler(local_image, local_double_image, lines_per_proc, XSIZE);
    free(local_image);
    //Do processing end

    if (rank == 0) {
        uchar *double_image = calloc(XSIZE * 2 * YSIZE * 2 * 3, 1);

        //Gathering the processed image parts from each process
        MPI_Gather(local_double_image, data_per_proc * 4, MPI_UNSIGNED_CHAR, double_image, data_per_proc * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        //saving image to file
        savebmp("after.bmp", double_image, XSIZE * 2, YSIZE * 2);
        free(double_image);
    } else {
        //sending data back to process 0
        MPI_Gather(local_double_image, data_per_proc * 4, MPI_UNSIGNED_CHAR, NULL, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

	free(local_double_image);
	MPI_Finalize();
	return 0;
}