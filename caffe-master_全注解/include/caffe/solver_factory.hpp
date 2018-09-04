/**
 * @brief A solver factory that allows one to register solvers, similar to layer factory During runtime,
 *    registered solvers could be called by passing a SolverParameter protobuffer to the CreateSolver function:
 *     solver factory  允许注册一个solver，运行 layer factory 也是一样的，
 *     注册solver能通过SolverParameter 的protobuffer解释文件的CreateSolver函数来创建
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 *  There are two ways to register a solver. Assuming that we have a solver like:
 *  有两种方法来注册，假设我有一个solver是：
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end("MyAwesomeSolver" -> "MyAwesome").
 * 他的类型是C++的class name ，但是没有Solver在后面   ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++ file, add the following line:
 *   如果这个solver 将要由构造函数简单的创建，在你的c++文件，添加下面的命令
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the format of:
 *   或者，如果这个solver 将要由其他的creator函数创建，格式为：
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 * 然后你就可以注册这个创建函数，像：
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 * 注意 ， 每个求解类型只能注册一次
 */

/*  简要说明：slover是什么？solver是caffe中实现训练模型参数更新的优化算法，
 *  solver类派生出的类可以对整个网络进行训练。在caffe中有很多solver子类，即不同的优化算法，如随机梯度下降（SGD）。
	一个solver factory可以注册一个 solvers，运行时，
	注册过的solvers通过SolverRegistry::CreateSolver(param)来调用。（就是在caffe.cpp中创建solver调用的那个）
	caffe提供两种方法注册一个solver */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
 //Creator是一个函数指针类型，指向的函数的参数为SolverParameter类型
  //，返回类型为Solver<Dtype>*
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();	//静态变量

  // Adds a creator.
    // 添加一个creator
  static void AddCreator(const string& type, Creator creator);

  // Get a solver using a SolverParameter.
  //  用 SolverParameter. 得到一个solver  
  // 静态成员函数，在caffe.cpp里直接调用，返回Solver指针
  static Solver<Dtype>* CreateSolver(const SolverParameter& param);

  static vector<string> SolverTypeList();

 private:
  // Solver registry should never be instantiated - everything is done with its static variables.
  //   solver 注册表不应该被实例化--所有的事都用它的静态变量完成  
  // Solver registry不应该被实例化，因为所有的成员都是静态变量
  // 构造函数是私有的，所有成员函数都是静态的，可以通过类调用
  SolverRegistry();   {}

  static string SolverTypeListString();
};

template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,	 // 指针函数
                   Solver<Dtype>* (*creator)(const SolverParameter&));
};
/*
分别定义了SolverRegisterer这个模板类的float和double类型的static变量，这会去调用各自
的构造函数，而在SolverRegisterer的构造函数中调用了之前提到的SolverRegistry类的
AddCreator函数，这个函数就是将刚才定义的Creator_SGDSolver这个函数的指针存到
g_registry指向的map里面。
*/
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

/*
这个宏会定义一个名为Creator_SGDSolver的函数，这个函数即为Creator类型的指针指向的函数，
在这个函数中调用了SGDSolver的构造函数，并将构造的这个变量得到的指针返回，这也就是Creator
类型函数的作用：构造一个对应类型的Solver对象，将其指针返回。然后在这个宏里又调用了
REGISTER_SOLVER_CREATOR这个宏
*/
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
