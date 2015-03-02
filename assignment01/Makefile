CC=mpicc
CFLAGS=-std=c99

SOURCES=assignment01.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=assignment01

DEBUG_LOG=debug_output.txt
ARCHIVE=assignment01.tar.xz
LAUNCHER=launcher.sh

%.o: %.c $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

all: assignment01

assignment01: $(OBJECTS)
	$(CC) -o $(EXECUTABLE) $(OBJECTS)

clean:
	rm -rf $(EXECUTABLE) $(DEBUG_LOG) $(ARCHIVE) *.o

test: assignment01
	./$(LAUNCHER) 3 4 4 1 > $(DEBUG_LOG)

archive: test
	tar cJf $(ARCHIVE) $(SOURCES) $(DEBUG_LOG) $(LAUNCHER)
