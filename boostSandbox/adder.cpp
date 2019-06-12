#include <boost/python.hpp>
using namespace boost::python;

class Adder {
  private:
    double a,b;

  public:
    Adder(double ay, double be) {
      a=ay;
      b=be;
    }
    void setA(double ay) {
      a=ay;
    }
    void setB(double be) {
      b=be;
    }
    double add() {
      return a+b;
    }
    tuple getter() { //tuple and make_tuple are in boost::python
      return make_tuple(a,b);
    }
};


BOOST_PYTHON_MODULE(adder)
{
  class_<Adder>("Adder", init<double, double>())
    .def("setA", &Adder::setA)
    .def("setB", &Adder::setB)
    .def("add", &Adder::add)
    .def("getter", &Adder::getter)
    ;
}
