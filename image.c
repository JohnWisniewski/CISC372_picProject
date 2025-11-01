#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>   // for sysconf
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---- helpers ----
static inline uint8_t clamp_u8(double v) {
    if (v < 0.0) return 0;
    if (v > 255.0) return 255;
    return (uint8_t)(v + 0.5); // round to nearest
}

// An array of kernel matrices to be used for image convolution.
// The indexes match the enumeration from the header file.
static const double algorithms[][3][3] = {
    { { 0,-1, 0},{-1, 4,-1},{ 0,-1, 0} },           // EDGE
    { { 0,-1, 0},{-1, 5,-1},{ 0,-1, 0} },           // SHARPEN
    { {1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0} }, // BLUR
    { {1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16} }, // GAUSE_BLUR (Gaussian 3x3)
    { {-2,-1,0},{-1,1,1},{0,1,2} },                  // EMBOSS
    { { 0, 0, 0},{ 0, 1, 0},{ 0, 0, 0} }             // IDENTITY
};

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

// ---- threaded worker ----
typedef struct {
    Image* src;
    Image* dst;
    Matrix kernel;  // now using Matrix type which includes const
    int row_start; // inclusive
    int row_end;   // exclusive
} ThreadJob;

void* worker(void* arg) {
    ThreadJob* job = (ThreadJob*)arg;
    Image* src = job->src;
    Image* dst = job->dst;

    for (int row = job->row_start; row < job->row_end; ++row) {
        for (int x = 0; x < src->width; ++x) {
            for (int bit = 0; bit < src->bpp; ++bit) {
                dst->data[Index(x, row, src->width, bit, src->bpp)] =
                    getPixelValue(src, x, row, bit, job->kernel);
            }
        }
    }
    return NULL;
}

// Usage helper
int Usage(){
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

// Decide thread count: THREADS env, else cores, else 4
static int decide_thread_count(void) {
    char* env = getenv("THREADS");
    if (env) {
        int t = atoi(env);
        if (t >= 1 && t <= 1024) return t;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 4;
    if (n > 64) n = 64; // sanity cap
    return (int)n;
}

int main(int argc,char** argv){
    long t1,t2;
    t1 = time(NULL);

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

    destImage.bpp    = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width  = srcImage.width;
    destImage.data   = malloc((size_t)destImage.width * destImage.height * destImage.bpp);
    if (!destImage.data) {
        fprintf(stderr, "Out of memory\n");
        stbi_image_free(srcImage.data);
        return -1;
    }

    // ---- launch threads over disjoint row ranges ----
    int T = decide_thread_count();
    if (T > destImage.height) T = destImage.height; // no point having more threads than rows

    pthread_t* tids = malloc(sizeof(pthread_t) * T);
    ThreadJob* jobs = malloc(sizeof(ThreadJob) * T);
    if (!tids || !jobs) {
        fprintf(stderr, "Out of memory (threads)\n");
        free(tids); free(jobs);
        stbi_image_free(srcImage.data);
        free(destImage.data);
        return -1;
    }

    int rows = destImage.height;
    int base = rows / T, rem = rows % T;
    int cur = 0;
    for (int i = 0; i < T; ++i) {
        int take = base + (i < rem ? 1 : 0);
    jobs[i].src = &srcImage;
    jobs[i].dst = &destImage;
    jobs[i].kernel = &algorithms[type][0];
    jobs[i].row_start = cur;
    jobs[i].row_end   = cur + take;
        cur += take;
        pthread_create(&tids[i], NULL, worker, &jobs[i]);
    }

    for (int i = 0; i < T; ++i) pthread_join(tids[i], NULL);
    free(tids);
    free(jobs);

    // ---- write output ----
    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp,
                   destImage.data, destImage.bpp * destImage.width);

    stbi_image_free(srcImage.data);
    free(destImage.data);

    t2 = time(NULL);
    printf("Took %ld seconds using %d thread(s)\n", t2 - t1, T);
    return 0;
}
