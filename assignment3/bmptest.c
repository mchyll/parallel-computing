#include <stdio.h>
#include "libs/bitmap.h"

int main(int argc, char **argv) {
    bmpImage* img = newBmpImage(10, 10);

    img->data[0][0] = (pixel) {255, 0, 0};
    img->data[0][1] = (pixel) {0, 255, 0};
    img->data[0][2] = (pixel) {0, 0, 255};

    img->data[1][0] = (pixel) {255, 0, 255};
    img->data[2][0] = (pixel) {255, 255, 0};
    img->data[3][0] = (pixel) {255, 255, 255};

    // Write the image back to disk
    if (saveBmpImage(img, "img.bmp") != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", "img.bmp");
        freeBmpImage(img);
        return 1;
    }

    return 0;
}
