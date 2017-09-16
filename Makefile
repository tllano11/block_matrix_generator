CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall
DEBUG = -g -D DEBUG
TARGET = matrix_generator

all: $(TARGET)

debug: CXXFLAGS += $(DEBUG)
debug: all

$(TARGET): $(TARGET).o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TARGET).o

$(TARGET).o: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp

clean:
	rm -f *~ *.o $(TARGET)
