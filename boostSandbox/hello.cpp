#include <Python.h>
#include <string>

std::string greet(){
  return "hello, world";
}

#include <boost/python.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(hello)
{
    def("greet",greet);
}
