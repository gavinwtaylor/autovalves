BOOST_ROOT := $(WORKDIR)/boost
PYTHON_ROOT := /gpfs/pkgs/mhpcc/anaconda3-5.0.1
PYTHON_INC := $(PYTHON_ROOT)/include/python3.6m
PYTHON_LIB_LOCATION := $(PYTHON_ROOT)/lib
BOOST_INC := $(BOOST_ROOT)/include
BOOST_LIB_LOCATION := $(BOOST_ROOT)/lib
BOOST_LIB_FILE := boost_python36

CC := gcc

CFLAGS := -c -fPIC
CInc := -I$(BOOST_INC) -I$(PYTHON_INC)

CLinkFlags = -shared -Wl,-soname,$@ -Wl,-rpath -L$(BOOST_LIB_LOCATION) -L$(PYTHON_LIB_LOCATION) -lpython3.6m -l$(BOOST_LIB_FILE)

all: hello.so world.so

world.so: world.o

hello.so: hello.o

%.so: %.o
	$(CC) $^ $(CLinkFlags) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) $(CInc) $^

clean:
	rm *.o *.so
