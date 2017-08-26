CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall
DEBUG = -g -D DEBUG
TARGET = main

all: $(TARGET)

debug: CXXFLAGS += $(DEBUG)
debug: all

$(TARGET): $(TARGET).o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TARGET).o

$(TARGET).o:
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp

clean:
	rm -f *~ *.o $(TARGET)
