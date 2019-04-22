CC=nvcc
TARGET=matrix_mul

.PHONY: run, clean
all:
	$(CC) -o $(TARGET).o $(TARGET).cu 
run:
	nvprof ./$(TARGET).o
clean:
	rm -rf *.o


