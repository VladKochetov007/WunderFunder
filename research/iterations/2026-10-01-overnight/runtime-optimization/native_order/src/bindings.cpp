// Python bindings (pybind11) for the streaming C++ engine: module `wnn_cpp`.
//
//   import wnn_cpp, numpy as np
//   w = dict(np.load("weights_0.npz"))
//   ens = wnn_cpp.Ensemble([w], [1.0])
//   ens.reset(); y = ens.step(row_float32_112)          # -> np.ndarray (2,)
//   Y = wnn_cpp.Predictor(w).run(x_float32_T_by_112)     # whole sequence, for tests
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "wnn/model.hpp"

namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

wnn::TransitionConfig to_config(const py::object& obj) {
  wnn::TransitionConfig c;
  if(obj.is_none()) return c;
  py::dict d=py::cast<py::dict>(obj);
  if(py::cast<std::string>(d["schema"])!="transhead01-68-v1") throw std::invalid_argument("feature schema");
  const std::string arm=py::cast<std::string>(d["arm"]);
  if(arm!="F" && arm!="T") throw std::invalid_argument("feature arm");
  py::dict n=py::cast<py::dict>(d["nuisance"]);
  using DArray=py::array_t<double,py::array::c_style|py::array::forcecast>;
  DArray centers=py::cast<DArray>(n["centers"]), pi=py::cast<DArray>(n["pi"]), R=py::cast<DArray>(n["R"]);
  if(centers.ndim()!=1 || centers.shape(0)!=4 || pi.ndim()!=1 || pi.shape(0)!=16 || R.ndim()!=2 || R.shape(0)!=16 || R.shape(1)!=16)
    throw std::invalid_argument("feature nuisance dimensions");
  std::copy(centers.data(),centers.data()+4,c.centers.begin()); std::copy(pi.data(),pi.data()+16,c.pi.begin());
  std::copy(R.data(),R.data()+256,c.prior.begin()); c.enabled=true; c.conditional=arm=="T"; c.validate(); return c;
}

wnn::Weights to_weights(const py::dict& d, const py::object& config=py::none()) {
  wnn::WeightMap m;
  for (auto item : d) {
    Array a = py::cast<Array>(item.second);
    wnn::Tensor t;
    t.data.assign(a.data(), a.data() + a.size());
    for (py::ssize_t i = 0; i < a.ndim(); ++i) t.shape.push_back(static_cast<int>(a.shape(i)));
    m.emplace(py::cast<std::string>(item.first), std::move(t));
  }
  return wnn::Weights::from(m,to_config(config));
}

const float* checked_row(const Array& row) {
  if (row.ndim() != 1 || row.shape(0) != wnn::kRaw)
    throw std::invalid_argument("row must be a float32 vector of length 112");
  return row.data();
}

template <class Engine>
py::array_t<float> run_sequence(Engine& e, const Array& x) {
  if (x.ndim() != 2 || x.shape(1) != wnn::kRaw) throw std::invalid_argument("x must be (T, 112)");
  const auto T = x.shape(0);
  py::array_t<float> out({T, static_cast<py::ssize_t>(2)});
  auto o = out.mutable_unchecked<2>();
  e.reset();
  float y[2];
  for (py::ssize_t t = 0; t < T; ++t) {
    e.step(x.data() + t * wnn::kRaw, y);
    o(t, 0) = y[0];
    o(t, 1) = y[1];
  }
  return out;
}

template <class Engine>
py::array_t<float> step_row(Engine& e, const Array& row) {
  py::array_t<float> out(2);
  e.step(checked_row(row), out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(wnn_cpp, m) {
  m.doc() = "Streaming GRU inference engine (C++ twin of wnn.nn.RNNModel + event features)";
  m.attr("N_RAW") = wnn::kRaw;
  m.attr("N_INPUTS") = wnn::kInputs;
  m.attr("HALF_SUPPORTED") = wnn::kernels::half_supported();

  py::class_<wnn::TransitionFeatures>(m,"TransitionFeatures")
      .def(py::init([](const py::dict& cfg){return wnn::TransitionFeatures(to_config(cfg));}))
      .def("reset",&wnn::TransitionFeatures::reset)
      .def("update",[](wnn::TransitionFeatures& f,const Array& raw){
        if(raw.ndim()!=1 || raw.shape(0)!=4) throw std::invalid_argument("expected raw4");
        py::array_t<float> out(wnn::kTransitionWidth); const float* got=f.update(raw.data());
        std::copy(got,got+wnn::kTransitionWidth,out.mutable_data()); return out;
      });

  py::class_<wnn::Predictor>(m, "Predictor")
      .def(py::init([](const py::dict& d, bool fast, bool half, bool int8, const py::object& config) {
             return wnn::Predictor(to_weights(d,config), fast, half, int8);
           }),
           py::arg("weights"), py::arg("fast_activations") = true, py::arg("half_weights") = false,
           py::arg("int8_weights") = false, py::arg("feature_config") = py::none())
      .def("reset", &wnn::Predictor::reset)
      .def("step", [](wnn::Predictor& p, const Array& row) { return step_row(p, row); }, py::arg("row"))
      .def("run", [](wnn::Predictor& p, const Array& x) { return run_sequence(p, x); }, py::arg("x"),
           "Reset, then predict every row of one sequence (T, 112) -> (T, 2).")
      .def_property_readonly("hidden", &wnn::Predictor::hidden)
      .def_property_readonly("half_weights", &wnn::Predictor::half)
      .def_property_readonly("int8_weights", &wnn::Predictor::int8)
      .def_property_readonly("online_norm", &wnn::Predictor::online_norm);

  py::class_<wnn::Ensemble>(m, "Ensemble")
      .def(py::init([](const std::vector<py::dict>& members, const std::vector<float>& coefs, bool fast, bool half,
                       bool int8, const py::object& configs) {
             if (members.size() != coefs.size() || members.empty())
               throw std::invalid_argument("need one coefficient per member and at least one member");
             wnn::Ensemble e;
             py::list cfgs;
             if(!configs.is_none()) {cfgs=py::cast<py::list>(configs); if(py::len(cfgs)!=members.size()) throw std::invalid_argument("one feature config per member");}
             for (std::size_t i = 0; i < members.size(); ++i)
               e.add(to_weights(members[i],configs.is_none()?py::none():py::reinterpret_borrow<py::object>(cfgs[i])),coefs[i],fast,half,int8);
             return e;
           }),
           py::arg("members"), py::arg("coefs"), py::arg("fast_activations") = true, py::arg("half_weights") = false,
           py::arg("int8_weights") = false, py::arg("feature_configs") = py::none())
      .def("reset", &wnn::Ensemble::reset)
      .def("step", [](wnn::Ensemble& e, const Array& row) { return step_row(e, row); }, py::arg("row"))
      .def("run", [](wnn::Ensemble& e, const Array& x) { return run_sequence(e, x); }, py::arg("x"))
      .def("__len__", &wnn::Ensemble::size);
}
