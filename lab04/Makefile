CUDACC=nvcc
CUDACFLAGS=-O3 -std=c++11 --ptxas-options=-v
CUDALDFLAGS=

SOURCES=reduction_sum.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=reduction_sum

.SUFFIXES:

.SUFFIXES: .cu .o

.cu.o: $(SOURCES)
	$(CUDACC) -c -o $@ $< $(CUDACFLAGS)

all: lab04

lab04: $(OBJECTS)
	$(CUDACC) $(CUDALDFLAGS) -o $(EXECUTABLE) $(OBJECTS)

clean: 
	rm -rf $(EXECUTABLE) *.o

