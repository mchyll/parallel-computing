#include <stdio.h>
#include "libs/bitmap.h"

bmpImageChannel* loadImage(char* filename) {
    bmpImage* image = newBmpImage(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
    }

    if (loadBmpImage(image, filename) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", filename);
        freeBmpImage(image);
        return NULL;
    }

    // Create a single color channel image. It is easier to work just with one color
    bmpImageChannel* imageChannel = newBmpImageChannel(image->width, image->height);
    if (imageChannel == NULL) {
        fprintf(stderr, "Could not allocate new image channel!\n");
        freeBmpImage(image);
        return NULL;
    }

    // Extract from the loaded image an average over all colors - nothing else than
    // a black and white representation
    // extractImageChannel and mapImageChannel need the images to be in the exact
    // same dimensions!
    // Other prepared extraction functions are extractRed, extractGreen, extractBlue
    if (extractImageChannel(imageChannel, image, extractAverage) != 0) {
        fprintf(stderr, "Could not extract image channel!\n");
        freeBmpImage(image);
        freeBmpImageChannel(imageChannel);
        return NULL;
    }

    return imageChannel;
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Supply two images to compare\n");
        return 0;
    }

    printf("Loading %s and %s for comparison\n", argv[1], argv[2]);

    bmpImageChannel* img1 = loadImage(argv[1]);
    if (img1 == NULL) {
        printf("Couldn't load %s\n", argv[1]);
        return 1;
    }

    bmpImageChannel* img2 = loadImage(argv[2]);
    if (img2 == NULL) {
        printf("Couldn't load %s\n", argv[2]);
        return 1;
    }

    if (img1->width != img2->width || img1->height != img2->height) {
        printf("Dimensions differ. Img1: %d x %d, img2: %d x %d\n", img1->width, img1->height, img2->width, img2->height);
        return 1;
    }

    bmpImage* diff = newBmpImage(img1->width, img1->height);
    unsigned char d;
    for (int i = 0; i < img1->width*img1->height; ++i) {
        d = (img1->rawdata[i] - img2->rawdata[i]);
        if (d < 0) d = -d;
        if (d > 255) d = 255;
        diff->rawdata[i].r = d;
        diff->rawdata[i].g = d;
        diff->rawdata[i].b = d;
    }

    // Write the diff image back to disk
    if (saveBmpImage(diff, "diff.bmp") != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", "diff.bmp");
        freeBmpImage(diff);
        return 1;
    }

    return 0;
}