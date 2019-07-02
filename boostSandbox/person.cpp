#include <iostream>
#include <Python.h>
#include <string>
#include <boost/python.hpp>
using namespace boost::python;
using namespace std;

class Person
{
  private:
      string name;
      int age;

  public: 
      Person(string n, int a)
      {
        name = n;
        age = a;
      }

      void eat(string food1)
      {
        cout << name << " is eating " << food1 << endl;
      }

      void printAge()
      {
        cout << name << " is " << age << " years old." << endl;
      }
};

BOOST_PYTHON_MODULE(person)
{
  class_<Person>("Person",init<string, int>())
    .def("eat", &Person::eat)
    .def("printAge", &Person::printAge)
    ;
}




