CC=gcc
CFLAGS=-fopenmp -std=c99 -O3
LDFLAGS=-fopenmp

SOURCES=assignment02.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=assignment02

DEBUG_LOG=debug_output.txt
ARCHIVE=assignment02.tar.xz

%.o: %.c $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

all: assignment02

assignment02: $(OBJECTS)
	$(CC) $(LDFLAGS) -o $(EXECUTABLE) $(OBJECTS)

clean:
	rm -rf $(EXECUTABLE) $(DEBUG_LOG) $(ARCHIVE) *.o

test: assignment02
	./$(EXECUTABLE) 3 4 4 1 > $(DEBUG_LOG)

archive: test
	tar cJf $(ARCHIVE) $(SOURCES) $(DEBUG_LOG) $(LAUNCHER)
