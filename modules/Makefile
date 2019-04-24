CC=nvcc
TARGET=singleIter

.PHONY: run, clean
all:
	$(CC) -o $(TARGET).o $(TARGET).cu 
run:
	nvprof ./$(TARGET).o
clean:
	rm -rf *.o


