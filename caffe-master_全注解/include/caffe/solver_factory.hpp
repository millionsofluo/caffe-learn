/**
 * @brief A solver factory that allows one to register solvers, similar to layer factory During runtime,
 *    registered solvers could be called by passing a SolverParameter protobuffer to the CreateSolver function:
 *     solver factory  ����ע��һ��solver������ layer factory Ҳ��һ���ģ�
 *     ע��solver��ͨ��SolverParameter ��protobuffer�����ļ���CreateSolver����������
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 *  There are two ways to register a solver. Assuming that we have a solver like:
 *  �����ַ�����ע�ᣬ��������һ��solver�ǣ�
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end("MyAwesomeSolver" -> "MyAwesome").
 * ����������C++��class name ������û��Solver�ں���   ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++ file, add the following line:
 *   ������solver ��Ҫ�ɹ��캯���򵥵Ĵ����������c++�ļ���������������
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the format of:
 *   ���ߣ�������solver ��Ҫ��������creator������������ʽΪ��
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 * Ȼ����Ϳ���ע�����������������
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 * ע�� �� ÿ���������ֻ��ע��һ��
 */

/*  ��Ҫ˵����slover��ʲô��solver��caffe��ʵ��ѵ��ģ�Ͳ������µ��Ż��㷨��
 *  solver��������������Զ������������ѵ������caffe���кܶ�solver���࣬����ͬ���Ż��㷨��������ݶ��½���SGD����
	һ��solver factory����ע��һ�� solvers������ʱ��
	ע�����solversͨ��SolverRegistry::CreateSolver(param)�����á���������caffe.cpp�д���solver���õ��Ǹ���
	caffe�ṩ���ַ���ע��һ��solver */

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
 //Creator��һ������ָ�����ͣ�ָ��ĺ����Ĳ���ΪSolverParameter����
  //����������ΪSolver<Dtype>*
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();	//��̬����

  // Adds a creator.
    // ���һ��creator
  static void AddCreator(const string& type, Creator creator);

  // Get a solver using a SolverParameter.
  //  �� SolverParameter. �õ�һ��solver  
  // ��̬��Ա��������caffe.cpp��ֱ�ӵ��ã�����Solverָ��
  static Solver<Dtype>* CreateSolver(const SolverParameter& param);

  static vector<string> SolverTypeList();

 private:
  // Solver registry should never be instantiated - everything is done with its static variables.
  //   solver ע���Ӧ�ñ�ʵ����--���е��¶������ľ�̬�������  
  // Solver registry��Ӧ�ñ�ʵ��������Ϊ���еĳ�Ա���Ǿ�̬����
  // ���캯����˽�еģ����г�Ա�������Ǿ�̬�ģ�����ͨ�������
  SolverRegistry();   {}

  static string SolverTypeListString();
};

template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,	 // ָ�뺯��
                   Solver<Dtype>* (*creator)(const SolverParameter&));
};
/*
�ֱ�����SolverRegisterer���ģ�����float��double���͵�static���������ȥ���ø���
�Ĺ��캯��������SolverRegisterer�Ĺ��캯���е�����֮ǰ�ᵽ��SolverRegistry���
AddCreator����������������ǽ��ղŶ����Creator_SGDSolver���������ָ��浽
g_registryָ���map���档
*/
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

/*
�����ᶨ��һ����ΪCreator_SGDSolver�ĺ��������������ΪCreator���͵�ָ��ָ��ĺ�����
����������е�����SGDSolver�Ĺ��캯���������������������õ���ָ�뷵�أ���Ҳ����Creator
���ͺ��������ã�����һ����Ӧ���͵�Solver���󣬽���ָ�뷵�ء�Ȼ������������ֵ�����
REGISTER_SOLVER_CREATOR�����
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
