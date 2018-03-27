/*  ==========================================================================================
    Author: Leonardo Citraro
    Institution: Computer Vision Laboratory - École polytechnique fédérale de Lausanne
    Description: Boost-Python wrapper for POM core functionalities
    
    Article: F. Fleuret, J. Berclaz, R. Lengagne and P. Fua, 
             Multi-Camera People Tracking with a Probabilistic Occupancy Map, 
             IEEE Transactions on Pattern Analysis and Machine Intelligence, 
             Vol. 30, Nr. 2, pp. 267 - 282, February 2008.
    ==========================================================================================
    Copyright (C) 2018  École polytechnique fédérale de Lausanne

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    ==========================================================================================
*/
#include <boost/python/numpy.hpp>
#include "core.hpp"

using namespace std;
//using namespace core;
namespace p = boost::python;
namespace np = boost::python::numpy;

/** 
  *   @brief  This function deletes dynamically allocated memory 
  *           from python side when the data is no loguer required.  
  */ 
template<typename T>
inline void CapsuleDestructor(PyObject *ptr){
    T *tmp = reinterpret_cast<T*>(PyCapsule_GetPointer(ptr, NULL));
    if(tmp){
        delete tmp;
    }
}

template<typename T>
inline np::ndarray make_ndarray(T* const ptr, const int H, const int W=0){

    // figuring out the size of the array
    int L;    
    if(H <= 0){
        L = W;
    }else if(W <= 0){
        L = H;
    }else{
        L = H*W;
    }

    // The object owner is responsible for deleting the dynamic array from the python side
    // This is the trick to prevent memory leaks! 
    p::object owner(p::handle<>((PyCapsule_New((void*)ptr, NULL, 
                                (PyCapsule_Destructor)&CapsuleDestructor<core::scalar_t>))));

    // Creates a boost numpy array, this is the actual object we return to python 
    np::ndarray py_array = np::from_data(ptr, 
                                         np::dtype::get_builtin<core::scalar_t>(),
                                         p::make_tuple(L),
                                         p::make_tuple(sizeof(core::scalar_t)),
                                         owner); 

    // in the case the returned array is 2D (H,W) instad of a 1D (L,)
    if(H > 0 && W > 0){
        py_array = py_array.reshape(p::make_tuple(H,W));
    }

    return py_array;
}

core::scalar_t sum(np::ndarray py_array){
    /// Parameters
    /// ----------
    /// py_array: 2D numpy array (H,W) [core::scalar_t]
    ///     Array
    ///
    /// Return
    /// ------
    /// Sum [core::scalar_t]

    int H = py_array.shape(0);
    int W = py_array.shape(1);
    int L = H*W;
    const core::scalar_t* array = reinterpret_cast<const core::scalar_t*>(py_array.get_data());
    return core::sum(array, L);
}

core::scalar_t sum_mask(np::ndarray py_array, np::ndarray py_mask){
    /// Parameters
    /// ----------
    /// py_array: 2D numpy array (H,W) [core::scalar_t]
    ///     Array
    /// py_mask: 2D numpy array [core::scalar_t]
    ///     Array
    ///
    /// Return
    /// ------
    /// Sum [core::scalar_t]

    int H = py_array.shape(0);
    int W = py_array.shape(1);
    int L = H*W;
    const core::scalar_t* array = reinterpret_cast<const core::scalar_t*>(py_array.get_data());
    const core::scalar_t* mask = reinterpret_cast<const core::scalar_t*>(py_mask.get_data());
    return core::sum_mask(array, mask, L);
}

np::ndarray sum_rows(np::ndarray py_array){
    /// Column-wise sum (sum of rows).
    ///
    /// Parameters
    /// ----------
    /// py_array: 2D numpy array (H,W) [core::scalar_t]
    ///     Array
    ///
    /// Return
    /// ------
    /// Sum : 1D numpy array (W,)  [core::scalar_t]

    int H = py_array.shape(0);
    int W = py_array.shape(1);
    const core::scalar_t** array = new const core::scalar_t*[H];

    for(int h=0; h<H; ++h){
	    np::ndarray temp = p::extract<np::ndarray>(py_array[h]);
        array[h] = reinterpret_cast<const core::scalar_t*>(temp.get_data());
    }

    core::scalar_t* const sum = core::sum_rows(array, H, W);
    np::ndarray py_sum = make_ndarray(sum, 1, W);

    return py_sum;
}

np::ndarray sum_cols(np::ndarray py_array){
    /// Row-wise sum (sum of columns).
    ///
    /// Parameters
    /// ----------
    /// py_array: 2D numpy array (H,W) [core::scalar_t]
    ///     Array
    ///
    /// Return
    /// ------
    /// Sum : 1D numpy array (H,)  [core::scalar_t]

    int H = py_array.shape(0);
    int W = py_array.shape(1);
    const core::scalar_t** array = new const core::scalar_t*[H];

    for(int h=0; h<H; ++h){
	    np::ndarray temp = p::extract<np::ndarray>(py_array[h]);
        array[h] = reinterpret_cast<const core::scalar_t*>(temp.get_data());
    }

    core::scalar_t* const sum = core::sum_cols(array, H, W);
    np::ndarray py_sum = make_ndarray(sum, 1, W);

    return py_sum;
}

np::ndarray compute_A_(p::tuple view_shape, np::ndarray py_rectangles, np::ndarray py_q){
    /// Computes part of equations (35) for one view/camera passing from  
    /// equations (31),(32),(33) and (34) of the paper. 
    ///
    /// Parameters
    /// ----------
    /// view_shape: tuple (2,)
    ///     View size/shape
    /// py_rectangles : 1D numpy array (4*K,) [int32]
    ///     Containes the points defining the rectangles 
    ///     i.e. numpy.int32([ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...])
    ///     If a rectangle is not visible, ymin, ymax, xmin and xmax must be set to -1.
    /// py_q : 1D numpy array (K,) [core::scalar_t]
    ///     Probability of presence
    ///
    /// Return
    /// ------
    /// A_ : 2D numpy array (view_shape) [core::scalar_t]

    int H = p::extract<int>(view_shape[0]);
    int W = p::extract<int>(view_shape[1]);
    int n_positions = p::len(py_q);
    const int* rectangles = reinterpret_cast<const int*>(py_rectangles.get_data());
    const core::scalar_t* q = reinterpret_cast<const core::scalar_t*>(py_q.get_data());

    core::scalar_t* const A = core::compute_A_(H, W, n_positions, rectangles, q);
    np::ndarray py_A = make_ndarray(A, H, W);

    return py_A;
}

np::ndarray integral_image(np::ndarray py_image){
    /// Computes the integral image/array
    ///
    /// Parameters
    /// ----------
    /// py_image: 2D numpy array (H,W) [core::scalar_t]
    ///     Image
    /// 
    /// Return
    /// ------
    /// Integral image : 2D numpy array (H,W) [core::scalar_t]

    int H = py_image.shape(0);
    int W = py_image.shape(1);
    int L = H*W;

    const core::scalar_t* image = reinterpret_cast<const core::scalar_t*>(py_image.get_data());
    core::scalar_t* const integral = core::integral_image(image, H, W);
    np::ndarray py_integral = make_ndarray(integral, H, W);
   
    return py_integral;
}

np::ndarray integral_image_mask(np::ndarray py_image, np::ndarray py_mask){
    /// Computes the integral image/array with mask
    ///
    /// Parameters
    /// ----------
    /// py_image: 2D numpy array (H,W) [core::scalar_t]
    ///     Image
    /// py_mask: 2D numpy array (H,W) [core::scalar_t]
    ///     Mask
    /// 
    /// Return
    /// ------
    /// Integral image : 2D numpy array (H,W) [core::scalar_t]

    int H = py_image.shape(0);
    int W = py_image.shape(1);
    int L = H*W;

    const core::scalar_t* image = reinterpret_cast<const core::scalar_t*>(py_image.get_data());
    const core::scalar_t* mask = reinterpret_cast<const core::scalar_t*>(py_mask.get_data());
    core::scalar_t* const integral = core::integral_image_mask(image, mask, H, W);
    np::ndarray py_integral = make_ndarray(integral, H, W);
   
    return py_integral;
}

core::scalar_t integral_sum(np::ndarray py_integral, 
                      const int ymin, const int ymax, const int xmin, const int xmax){
    /// Computes the sum of a rectangular region 
    /// in the image using the integral image. 
    /// 
    /// Parameters
    /// ----------
    /// integral: 2D numpy array (H,W) [core::scalar_t]
    ///     Integral image
    /// (ymin, xmin):
    ///     Upper left corner of the rectangle/box
    /// (ymax, xmax):
    ///     Bottom right corner of the rectangle/box
    /// 
    /// Return
    /// ------
    /// sum of the element in the region [core::scalar_t]

    int W = py_integral.shape(1);
    const core::scalar_t* integral = reinterpret_cast<const core::scalar_t*>(py_integral.get_data());

    return core::integral_sum(integral, W, ymin, ymax, xmin, xmax);
}

np::ndarray compute_psi_diff(np::ndarray py_Ai, np::ndarray py_BAi, const core::scalar_t lAl, 
                             const core::scalar_t lBxAl, const core::scalar_t lBl, 
                             np::ndarray py_rectangles, np::ndarray py_q){
    /// Computes part of equations (35) for one view/camera passing from  
    /// equations (31),(32),(33) and (34) of the paper.    
    /// 
    /// Parameters
    /// ----------
    /// py_Ai : 2D numpy array (H,W) [core::scalar_t]
    ///     Integral image (1-A)
    /// py_BAi : 2D numpy array (H,W) [core::scalar_t]
    ///     Integral image Bx(1-A) 
    /// lAl : value [core::scalar_t]
    ///      
    /// lBxAl : value [core::scalar_t]
    ///
    /// lBl : value [core::scalar_t]
    ///
    /// py_rectangles : 1D numpy array (4*K, ) [int32]
    ///     Rectangles for all the views/cameras
    ///     in the format [ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...].
    ///     In the case the rectangle is not visible set ymin,ymax,xmin,xmax to -1
    /// py_q : 1D numpy array (K,) [core::scalar_t]
    ///     Probability of presence
    /// 
    /// Return
    /// ------
    /// psi(B, A1) - psi(B, A0) 1D numpy array (K,) [core::scalar_t]

    int H = py_Ai.shape(0);
    int W = py_Ai.shape(1);
    int n_positions = py_q.shape(0);
    int const* rectangles   = reinterpret_cast<const int*>(py_rectangles.get_data());
    core::scalar_t const* q = reinterpret_cast<core::scalar_t const*>(py_q.get_data());
    core::scalar_t const* Ai = reinterpret_cast<core::scalar_t const*>(py_Ai.get_data());
    core::scalar_t const* BAi = reinterpret_cast<core::scalar_t const*>(py_BAi.get_data());

    core::scalar_t* const psi_diff = core::compute_psi_diff(H, W, Ai, BAi, lAl, lBxAl, lBl, rectangles, q, n_positions);

    np::ndarray py_psi_diff = make_ndarray(psi_diff, n_positions);

    return py_psi_diff;
}      

// This function is never used because my implementation of the sum in C++ is 
// slower then numpy.sum(). For this reason the final algorithm is mostly written in Python
// but with some C++ code. Have a look at the class Solver of the file pom.py

int solve(p::list py_B, np::ndarray py_q, p::list py_rectangles, const core::scalar_t prior, 
			const core::scalar_t sigma, const core::scalar_t step, const int max_iter, const core::scalar_t tol){
    /// The full detection algorithm.    
    /// 
    /// Parameters
    /// ----------
    /// py_B : list of 2D numpy array [(H,W),(H,W),..] [core::scalar_t]
    ///     Background substraction images for all the views/cameras
    /// py_q : 1D numpy array (K,) [core::scalar_t] !!!!Modified inplace!!!!!
    ///     Probability of presence
    /// py_rectangles : list of 1D numpy array [(4*K,),(4*K,),..] [int32]
    ///     Rectangles for all the views/cameras
    ///     in the format [ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...].
    ///     In the case the rectangle is not visible set ymin,ymax,xmin,xmax to -1
    /// prior : value [core::scalar_t]
    ///      Prior probability of presence
    /// sigma : value [core::scalar_t]
    ///     Quality of the background segmentation images
    /// step : value [core::scalar_t]
    ///     ~learning rate for the optimization
    /// max_iter : value [int32]
    ///     Maximum numer of iteration for the optimization
    /// tol : value [core::scalar_t]
    ///     tollerance for the optimization, if reached the functions returns
    /// 
    /// Return
    /// ------
    /// Number of iteration performed [int32]
    ///
    /// py_q is modified inplace

    int c;
    np::ndarray py_B0 = p::extract<np::ndarray>(py_B[0]);
    int H = py_B0.shape(0);
    int W = py_B0.shape(1);
    int n_cams = p::len(py_B);
    int n_positions = p::len(py_q);
    const core::scalar_t** B = new const core::scalar_t*[n_cams];
    const int** rectangles = new const int*[n_cams];

    for(c=0; c<n_cams; ++c){
	    np::ndarray temp1 = p::extract<np::ndarray>(py_B[c]);
        B[c] = reinterpret_cast<const core::scalar_t*>(temp1.get_data());

        np::ndarray temp2 = p::extract<np::ndarray>(py_rectangles[c]);
        rectangles[c] = reinterpret_cast<const int*>(temp2.get_data());
    }

    // we modify q inplace!
    core::scalar_t* q = reinterpret_cast<core::scalar_t*>(py_q.get_data());

    int exit_iter = core::solve(B, rectangles, q, n_cams, n_positions, H, W, prior, sigma, step, max_iter, tol);

    return exit_iter;

}

BOOST_PYTHON_MODULE(core) {   

    Py_Initialize();
    np::initialize();

    p::def("compute_A_", compute_A_);
    p::def("integral_image", integral_image);
    p::def("integral_image_mask", integral_image_mask);
    p::def("integral_sum", integral_sum);
    p::def("compute_psi_diff", compute_psi_diff);
    p::def("solve", solve);
    p::def("sum", sum);
    p::def("sum_mask", sum_mask);
    p::def("sum_rows", sum_rows);
    p::def("sum_cols", sum_cols);
};
