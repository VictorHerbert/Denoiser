# Compiler and flags
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
BLENDER = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

# Targets
TARGET = $(BUILD_DIR)/main

# Default target
all: $(TARGET)

run: $(TARGET)
	@mkdir -p build/sample
	@./$(TARGET) -r sample/cornell/32/Render.png -s build/sample/cornell32.png

test: $(TARGET)
	@mkdir -p build/test
	@./$(TARGET) -t

debug: $(TARGET)
	cuda-gdb -ex=run -ex=quit ./$(TARGET)

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck --show-backtrace=yes ./$(TARGET)

prof:
	@nsys profile -o build/prof ./$(TARGET) -t
	nsight-sys build/prof.nsys-rep

# Link main
$(TARGET): $(OBJ)
	@$(NVCC) $(CXXFLAGS_LK) -o $@ $^

# Compile rules with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	$(NVCC) $(CXXFLAGS) -c $< -o $@

render: scenes/cornell.blend
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 1
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 4
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 8
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 16
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 32
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 8192

.PHONY: render

# Clean
clean:
	@rm -rf $(BUILD_DIR)/*
	@mkdir -p $(BUILD_DIR)

# Include auto-generated dependency files
-include $(OBJ:.o=.d)

.PHONY: all clean run debug sanitize
