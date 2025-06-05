# Compiler and flags
#NVCC = /usr/local/cuda-12.8/bin/nvcc
NVCC = nvcc
CXX = $(NVCC)
CXXFLAGS_LK = -w -G -g -O0 -std=c++17 -arch=sm_75 -I./include
CXXFLAGS = $(CXXFLAGS_LK) -dc
LDFLAGS =  # Optional linker flags

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Source files
SRC = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ := $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Targets
TARGET = $(BUILD_DIR)/main

# Default target
all: $(TARGET)

run: $(TARGET)
	./$(TARGET) -r sample/cornell/32/Render.png

debug: $(TARGET)
	cuda-gdb -ex=run -ex=quit ./$(TARGET)

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck --show-backtrace=yes ./$(TARGET)

# Link main
$(TARGET): $(OBJ)
	$(NVCC) $(CXXFLAGS_LK) -o $@ $^

# Compile rules with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)/*
	rm -rf $(OUTPUT_DIR)/*
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OUTPUT_DIR)

# Include auto-generated dependency files
-include $(OBJ:.o=.d)

.PHONY: all clean
