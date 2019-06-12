#include <iostream>
#include <string>
#include <Python.h>

using namespace std;

class Rabbit
{
   public:

   //Data Members
   
   string rabbitname;
   int whiskers;
   bool cute;

   //Member Function
   void set_name(string name);
   string get_name(void);
};

   
   
   void Rabbit::set_name(string name){
     rabbitname=name;
   }
  
  string Rabbit::get_name(void){
    return "I am a bunny rabbit named " + rabbitname;
  }

  

#include <boost/python.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(rabbit)
{
  class_<Rabbit>("Rabbit")
   .def("set_name", &Rabbit::set_name)
   .def("get_name",&Rabbit::get_name)
   ;
}


  
