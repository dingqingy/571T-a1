CC=nvcc
TARGET=matrix_mul_shared

.PHONY: run, clean
all:
	$(CC) -o $(TARGET).o $(TARGET).cu 
run:
	nvprof ./$(TARGET).o
clean:
	rm -rf *.o


