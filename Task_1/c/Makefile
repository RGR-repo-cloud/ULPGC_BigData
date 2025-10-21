# Makefile for C Matrix Multiplication Benchmark

CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

# Source files
SOURCES = matrix_multiply.c matrix_utils.c stats_utils.c csv_writer.c matrix_benchmark.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = matrix_benchmark

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run