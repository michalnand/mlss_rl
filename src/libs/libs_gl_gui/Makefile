OUTPUT_NAME = gl_gui

INC+= -I.

SRCS+= $(shell find . -name '*.cpp')

CC = g++
CU = nvcc
LD = g++
OBJCOPY = objcopy
OBJDUMP = objdump
SWIG = swig3.0
SIZE = size
ARCHIVE = ar

CFLAGS = -O2 -Os -fPIC -s -Wall -Wextra -pedantic -std=c++11 $(INC)

PYTHONI = $(shell python3-config --cflags)
PYTHONL = $(shell python3-config --ldflags)

LD_OPENGLFLAGS = -lglut -lGL -lGLU -lX11
LD_COMMONFLAGS = -ljsoncpp

LDFLAGS = $(LD_OPENGLFLAGS) $(LD_COMMONFLAGS) -lm -lpthread -pthread -std=c++17 -lstdc++fs

OBJS = $(SRCS:.cpp=.o)
SRCS_SWIG+= $(shell find . -name '*.i')

OBJS_CSWIG  = $(SRCS_SWIG:.i=.cxx)
OBJS_SWIG 	= $(OBJS_CSWIG:.cxx=.oswig)

all: $(OBJS) $(OBJS_SWIG)
	$(ARCHIVE) rc $(OUTPUT_NAME).a $(OBJS)
	$(CC) -shared -fPIC $(PYTHONL) -o _$(OUTPUT_NAME).so $(OBJS) $(OBJS_EXTERN) $(OBJS_SWIG) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.cxx: %.i
	$(SWIG) -c++ -python $(INC) -o $@ $<

%.oswig: %.cxx
	$(CC) -c -fPIC $(INC) $(PYTHONI) $< -o $@



clean:
	rm $(OBJS) -f
	rm  $(OBJS_SWIG) -f
	rm $(OUTPUT_NAME).a -f
	rm _$(OUTPUT_NAME).so -f
