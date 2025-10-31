#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---- optional helper to keep values in [0,255] ----
static inline uint8_t clamp_u8(double v) {
    if (v < 0.0) return 0;
    if (v > 255.0) return 255;
    return (uint8_t)(v + 0.5);
}

// An array of kernel matrices to be used for image convolution.
// The indexes match the enumeration from the header file.
Matrix algorithms[] = {
    { { 0,-1, 0},{-1, 4,-1},{ 0,-1, 0} },           // EDGE
    { { 0,-1, 0},{-1, 5,-1},{ 0,-1, 0} },           // SHARPEN
    { {1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0} }, // BLUR
    { {1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16} }, // GAUSE_BLUR
    { {-2,-1,0},{-1,1,1},{0,1,2} },                  // EMBOSS
    { { 0, 0, 0},{ 0, 1, 0},{ 0, 0, 0} }             // IDENTITY
};

// Compute one pixel/channel via 3x3 convolution
uint8_t getPixelValue(Image* srcImage, int x, int y, int bit, Matrix algorithm){
    int px = x + 1, mx = x - 1, py = y + 1, my = y - 1;
    if (mx < 0) mx = 0;
    if (my < 0) my = 0;
    if (px >= srcImage->width)  px = srcImage->width - 1;
    if (py >= srcImage->height) py = srcImage->height - 1;

    double acc =
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[0][1]*srcImage->data[Index(x ,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][0]*srcImage->data[Index(mx,y ,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][1]*srcImage->data[Index(x ,y ,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][2]*srcImage->data[Index(px,y ,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][1]*srcImage->data[Index(x ,py,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];
    return clamp_u8(acc);
}

// Parallel convolution: split by rows (no overlap in writes)
void convolute(Image* srcImage, Image* destImage, Matrix algorithm){
    // outer loop parallelized; each thread writes disjoint rows -> no race
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < srcImage->height; ++row){
        for (int pix = 0; pix < srcImage->width; ++pix){
            for (int bit = 0; bit < srcImage->bpp; ++bit){
                destImage->data[Index(pix,row,srcImage->width,bit,srcImage->bpp)] =
                    getPixelValue(srcImage, pix, row, bit, algorithm);
            }
        }
    }
}

// Usage text
int Usage(void){
    printf("Usage: image_openmp <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

// Map CLI string to kernel
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

int main(int argc,char** argv){
    long t1 = time(NULL);

    stbi_set_flip_vertically_on_load(0);
    if (argc != 3) return Usage();

    char* fileName = argv[1];
    if (!strcmp(argv[1],"pic4.jpg") && !strcmp(argv[2],"gauss")){
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }
    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, destImage;
    srcImage.data = stbi_load(fileName, &srcImage.width, &srcImage.height, &srcImage.bpp, 0);
    if (!srcImage.data){
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    destImage.bpp = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width = srcImage.width;
    destImage.data = malloc((size_t)destImage.width * destImage.height * destImage.bpp);
    if (!destImage.data){
        fprintf(stderr, "Out of memory\n");
        stbi_image_free(srcImage.data);
        return -1;
    }

    // Run parallel convolution
    convolute(&srcImage, &destImage, algorithms[type]);

    // Write result
    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp,
                   destImage.data, destImage.bpp * destImage.width);

    stbi_image_free(srcImage.data);
    free(destImage.data);

    long t2 = time(NULL);
    int threads_used = 1;
    #ifdef _OPENMP
    threads_used = omp_get_max_threads();
    #endif
    printf("Took %ld seconds with OpenMP (%d thread(s) max)\n", t2 - t1, threads_used);
    return 0;
}
