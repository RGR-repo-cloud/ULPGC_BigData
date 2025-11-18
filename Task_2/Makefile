CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c99
INCLUDES = -Iinclude -Ibenchmarks
SRCDIR = src
BENCHDIR = benchmarks
OBJDIR = obj

# Source files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Benchmark files
BENCH_SRCS = $(wildcard $(BENCHDIR)/*.c)
BENCH_OBJS = $(BENCH_SRCS:$(BENCHDIR)/%.c=$(OBJDIR)/%.o)

# Targets
TARGET = matrix_benchmark
DEBUG_TARGET = matrix_benchmark_debug

.PHONY: all clean debug run

all: $(TARGET)

debug: CFLAGS = -Wall -Wextra -g -std=c99 -DDEBUG
debug: $(DEBUG_TARGET)

$(TARGET): $(OBJS) $(BENCH_OBJS) | $(OBJDIR)
	$(CC) $(OBJS) $(BENCH_OBJS) -o $@ -lm

$(DEBUG_TARGET): $(OBJS) $(BENCH_OBJS) | $(OBJDIR)
	$(CC) $(OBJS) $(BENCH_OBJS) -o $@ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: $(BENCHDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET) $(DEBUG_TARGET)

help:
	@echo "Available targets:"
	@echo "  all     - Build optimized version"
	@echo "  debug   - Build debug version"
	@echo "  run     - Build and run benchmark"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"