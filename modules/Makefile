CC=nvcc
TARGET=forward

.PHONY: run, clean
all:
	$(CC) -o $(TARGET).o $(TARGET).cu 
run:
	nvprof ./$(TARGET).o
clean:
	rm -rf *.o


