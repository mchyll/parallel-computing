
ITER=10

.PHONY: clean run runserial

main: libs/bitmap.c libs/bitmap.c main.cu
	nvcc libs/bitmap.c main.cu -o main

run: main
	./main before.bmp after.bmp -i $(ITER)

clean:
	rm -Rf *.o
	rm -Rf main
	rm -f after.bmp

# end
