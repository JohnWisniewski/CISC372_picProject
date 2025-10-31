# Step 2: pthreads
image_pthreads: image.c image.h
	gcc -g image.c -o image_pthreads -lm -pthread

# Step 3: OpenMP
image_openmp: image.c image.h
	gcc -g -fopenmp image.c -o image_openmp -lm

clean:
	rm -f image_pthreads image_openmp output.png