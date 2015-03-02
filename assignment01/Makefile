CC=mpicc
CFLAGS=-std=c99
OBJECTS=assignment01.o
EXECUTABLE=assignment01

all: assignment01

assignment01: $(OBJECTS)
	$(CC) -o $(EXECUTABLE) $(OBJECTS)

clean:
	rm -rf $(EXECUTABLE) *.o

test: assignment01
	mpirun -np 12 ./$(EXECUTABLE) 3 4 4 1
