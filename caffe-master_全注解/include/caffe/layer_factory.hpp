/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
   // 函数指针Creator，返回的是Layer<Dtype>类型的指针
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
    // CreatorRegistry是字符串与对应的Creator的映射
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();

  // Adds a creator.
    // 给定类型，以及函数指针，加入到注册表
  static void AddCreator(const string& type, Creator creator);

  // Get a layer using a LayerParameter.
    // 通过LayerParameter，返回特定层的实例智能指针
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param);

  static vector<string> LayerTypeList();

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
    // 私有构造函数，禁止实例化，所有功能都由静态函数完成，所以不需要实例化
  LayerRegistry();

  static string LayerTypeListString();
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&));
};
/*REGISTER_LAYER_CREATOR负责将创建层的函数放入LayerRegistry*/
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \
    
/* 大多数层创建的函数的生成宏REGISTER_LAYER_CLASS，可以看到宏函数比较简单的，
	将类型作为函数名称的一部分，这样就可以产生出一个创建函数，
	并将创建函数放入LayerRegistry。*/
		/*
 * 宏 REGISTER_LAYER_CLASS 为每个type生成了create方法，并和type一起注册到了LayerRegistry中
 * ，保存在一个map里面。
 */
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
