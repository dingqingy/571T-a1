CC=nvcc
TARGET=d_relu

.PHONY: run, clean
all:
	$(CC) -o $(TARGET).o $(TARGET).cu 
run:
	nvprof ./$(TARGET).o
clean:
	rm -rf *.o


