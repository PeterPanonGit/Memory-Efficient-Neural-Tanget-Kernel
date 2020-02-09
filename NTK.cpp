#include "NTK.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

// wrap as Python module
PYBIND11_MAKE_OPAQUE(std::vector<float>);

PYBIND11_MODULE(NtkInternal, m) {
    m.doc() = "NTK C++ plugin";

    // Expose float type matrix
    // Double type Matrix class can be created similarly but skipped here.
    // This MatrixF can be moved into python with zero copy using:
    // a = np.array(MatrixF_instance, copy = False)
    py::class_<Matrix<float>>(m, "MatrixF", py::buffer_protocol())
       .def(py::init<int, int>())
       .def(py::init<Matrix<float>&>())
       .def("value", &Matrix<float>::value)
       .def("nrow", &Matrix<float>::nrow)
       .def("ncol", &Matrix<float>::ncol)
       .def("set_val", &Matrix<float>::set_val)
       .def("set_zero", &Matrix<float>::set_zero)
       .def("copy", &Matrix<float>::copy)
       .def("print", &Matrix<float>::print)
       .def_buffer([](Matrix<float>& mat) -> py::buffer_info {
            return py::buffer_info(
                mat.data(),                               /* Pointer to buffer */
                sizeof(float),                            /* Size of one scalar */
                py::format_descriptor<float>::format(),   /* Python struct-style format descriptor */
                2,                                        /* Number of dimensions */
                { mat.nrow(), mat.ncol() },               /* Buffer dimensions */
                { sizeof(float) * mat.ncol(),             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

    // Expose float type vector. Double vector is skipped here.
    py::bind_vector<std::vector<float>>(m, "VectorF");

    m.def("Ntk", &Ntk<float>, "Calculate the embedding covariance matrix \
        and neural tangent kernel matrix");
}
