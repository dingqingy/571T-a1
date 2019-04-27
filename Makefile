CC=nvcc
TARGET1=simpleBackprop
TARGET2=sharedBackprop

.PHONY: part1, part2, run, clean
part1:
	$(CC) -o $(TARGET1).o $(TARGET1).cu 
part2:
	$(CC) -o $(TARGET2).o $(TARGET2).cu 
run1:
	nvprof ./$(TARGET1).o
run2:
	nvprof ./$(TARGET2).o
clean:
	rm -rf *.o


