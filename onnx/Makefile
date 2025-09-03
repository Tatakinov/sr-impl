CXXFLAGS=-g -O2 -Wall -std=c++20 -I . -I include $(shell pkg-config --cflags stb libonnxruntime)
LDFLAGS=-L . $(shell pkg-config --libs stb libonnxruntime)
OBJ=main.o
TARGET=sr_impl

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) $(TARGET) $(OBJ)
