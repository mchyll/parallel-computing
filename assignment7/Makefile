
ITER=10

.PHONY: clean run

main: libs/bitmap.c libs/bitmap.c main.cu
	nvcc libs/bitmap.c main.cu -o main
	# Add -arch=sm_60 -rdc=true if doing task 2

run: main
	./main before.bmp after.bmp -i $(ITER)

clean:
	rm -Rf *.o
	rm -Rf main
	rm -f after.bmp

# end
