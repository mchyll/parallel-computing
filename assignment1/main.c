#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

uchar* modifyImage(uchar* image);

int main() {
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	uchar* modifiedImage = modifyImage(image);

	savebmp("after.bmp", modifiedImage, XSIZE * 2, YSIZE * 2);
	free(image);
    free(modifiedImage);
	return 0;
}

uchar* modifyImage(uchar* image) {
    uchar* modifiedImage = calloc(4 * XSIZE * YSIZE * 3, 1); // Modified image is double the size

    // Loop through all pixels in the up-scaled image
    for (int x = 0; x < XSIZE * 2; ++x) {
        for (int y = 0; y < YSIZE * 2; ++y) {
            // Index of the first color component of the current pixel in the original image array
            int indexOriginal = ((int)(y / 2) * XSIZE + (int)(x / 2)) * 3;

            // Index of the first color component of the current pixel in the up-scaled image array
            int indexModified = (y * XSIZE * 2 + x) * 3;

            // Simple manipulation; swaps RGB color components around
            modifiedImage[indexModified] = image[indexOriginal + 1];
            modifiedImage[indexModified + 1] = image[indexOriginal + 2];
            modifiedImage[indexModified + 2] = image[indexOriginal];
        }
    }

    return modifiedImage;
}
