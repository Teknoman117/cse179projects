CUDACC=nvcc
CUDACFLAGS=-O3 -std=c++11 --ptxas-options=-v
CXXFLAGS=-fopenmp -O3 -std=c++11 -I/opt/cuda/include -I/usr/local/cuda/include
LDFLAGS=-lgomp

SOURCES_CUDA=task1.cu task2.cu task3.cu
SOURCES_CXX=utilities.cpp assignment03.cpp
EXECUTABLE=assignment03

OBJECTS_CUDA=$(SOURCES_CUDA:.cu=.o)
OBJECTS_CXX=$(SOURCES_CXX:.cpp=.o)

.SUFFIXES:

.SUFFIXES: .cu .cpp .o

.cu.o:
	$(CUDACC) -c -o $@ $< $(CUDACFLAGS)

.cpp.o:
	$(CXX) -c -o $@ $< $(CXXFLAGS)

all: task1 task2 task3

task1: $(OBJECTS_CUDA) $(OBJECTS_CXX)
	$(CUDACC) $(LDFLAGS) -o $(EXECUTABLE)_task1 task1.o $(OBJECTS_CXX)

task2: $(OBJECTS_CUDA) $(OBJECTS_CXX)
	$(CUDACC) $(LDFLAGS) -o $(EXECUTABLE)_task2 task2.o $(OBJECTS_CXX)

task3: $(OBJECTS_CUDA) $(OBJECTS_CXX)
	$(CUDACC) $(LDFLAGS) -o $(EXECUTABLE)_task3 task3.o $(OBJECTS_CXX)

clean:
	rm -rf $(EXECUTABLE)_* *.o
