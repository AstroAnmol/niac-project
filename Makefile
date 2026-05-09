# TODO: Move `libmongoclient.a` to /usr/local/lib so this can work on production servers
#
 
CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
TARGET := bin/runner
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -O3 -std=c++17 # -Wall
OS := $(shell uname -s)
ifneq ($(OS),Darwin)
	CFLAGS += -fopenmp
endif

INC := -I include -I include/boost_1_91_0

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $(CFLAGS) $^ -o $(TARGET) $(LIB)"; $(CC) $(CFLAGS) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

# Tests
test_detection: $(filter-out $(BUILDDIR)/main.o, $(OBJECTS)) test/test_detection.cpp
	@mkdir -p bin
	$(CC) $(CFLAGS) $(INC) test/test_detection.cpp $(filter-out $(BUILDDIR)/main.o, $(OBJECTS)) -o bin/test_detection $(LIB)

# Spikes
ticket:
 # $(CC) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket

.PHONY: clean
