#include <boost/python/numpy.hpp>
#include <iostream>
/*
bp::handle<> h(::PyCapsule_New((void *)data, NULL, (PyCapsule_Destructor)&destroyManagerCObject));
arr = np::from_data(data_, 
                    dtype, 
                    np_shape,
                    np_stride,
                    bp::object(h) 
                    );

inline void destroyManagerCObject(PyObject* obj) {
    double * b = reinterpret_cast<double*>(PyCapsule_GetPointer(obj, NULL) );
    if(b){ 
        delete [] b; 
    }
}
*/



using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

using scalar_t = double; // double is faster than float?????


template<typename T>
inline void CapsuleDestructor(PyObject *ptr){
    T *tmp = reinterpret_cast<T*>(PyCapsule_GetPointer(ptr, NULL));
    if(tmp){
        delete tmp;
    }
}

/**
 **  Make sure the pointer points the first element of the array !!!!!
 **/
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
    // This prevents memory leaks! 
    p::object owner(p::handle<>((PyCapsule_New((void*)ptr, NULL, 
                                (PyCapsule_Destructor)&CapsuleDestructor<scalar_t>))));

    // create a boost numpy array, this is the actual object we return to python 
    np::ndarray py_array = np::from_data(ptr, 
                                         np::dtype::get_builtin<scalar_t>(),
                                         p::make_tuple(L),
                                         p::make_tuple(sizeof(scalar_t)),
                                         owner); 

    // in the case the returned array is 2D (H,W) instad of a 1D (L,)
    if(H > 0 && W > 0){
        py_array = py_array.reshape(p::make_tuple(H,W));
    }

    return py_array;
}



scalar_t* const compute_A_impl(const int H, const int W, const int n_positions, const int* rectangles, const scalar_t* q){

    int i,j,k;
    scalar_t proba_absence;
    int ymin,ymax,xmin,xmax;
    int L = H*W;

    scalar_t* const A = new scalar_t[L];
    scalar_t* p_A = A;
    for(i=0; i<L; ++i){
        *(p_A++) = 1.0;   
    }  

    for(k=0; k<n_positions; ++k){
        ymin = *(rectangles++);
        ymax = *(rectangles++);
        xmin = *(rectangles++);
        xmax = *(rectangles++);
        if(ymin>-1){
            // we make sure to not go outside the image
            ymin = max(ymin, 0);
            ymax = min(ymax, H-1);
            xmin = max(xmin, 0);
            xmax = min(xmax, W-1);

            //cout << ymin << "," << ymax << "," << xmin << "," << xmax << "--" << *q << endl;

            proba_absence = 1.0 - *q;
            
            for(i=ymin; i<=ymax; ++i){
                p_A = A + i*W+xmin;
                for(j=xmin; j<=xmax; ++j){                    
                    *p_A *= proba_absence;
                    p_A++;
                }   
            } 
             
        } 
        q++;   
    }
        
    return A;
}

np::ndarray compute_A_(p::tuple view_shape, np::ndarray py_rectangles, np::ndarray py_q){

    /// Parameters
    /// ----------
    /// view_shape: tuple (2,)
    ///     View size/shape
    /// rectangles : 1D numpy array (4*K,) [int32]
    ///     Containes the points defining the rectangles 
    ///     i.e. numpy.int32([ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...])
    ///     If a rectangle is not visible, ymin, ymax, xmin and xmax must be set to 0.
    /// q : 1D numpy array (K,) [float32]
    ///     Probability of presence
    ///
    /// Return
    /// ------
    /// A_ : 2D numpy array (view_shape) [float32]

    int H = p::extract<int>(view_shape[0]);
    int W = p::extract<int>(view_shape[1]);
    int n_positions = p::len(py_q);
    const int* rectangles = reinterpret_cast<const int*>(py_rectangles.get_data());
    const scalar_t* q = reinterpret_cast<const scalar_t*>(py_q.get_data());

    scalar_t* const A = compute_A_impl(H, W, n_positions, rectangles, q);
    np::ndarray py_A = make_ndarray(A, H, W);

    return py_A;
}

scalar_t* const integral_image_impl(const scalar_t* p_image, const int H, const int W){
    /*
    Parameters
    ----------
    image: 2D numpy array (H,W) [float32]
        Image

    Return
    ------
    Integral image : 2D numpy array (H,W) [float32]
    */

    int i,j;
    int L = H*W;
    float sum = 0;
    scalar_t* const integral = new scalar_t[L];
    scalar_t* p_integral = integral;    

    // For the first row
    for(j=0; j<W; ++j){
        sum += *(p_image++);
        *(p_integral++) = sum;         
    } 
    
    // For the rest of the rows
    for(i=1; i<H; ++i){
        sum = 0;
        for(j=0; j<W; ++j){ 
            *(p_integral) = sum +*(p_integral-W) + *(p_image); 
            sum += *(p_image);
            p_integral++;
            p_image++;
        }    
    }
   
    return integral;
}

np::ndarray integral_image(np::ndarray py_image){
    /*
    Parameters
    ----------
    py_image: 2D numpy array (H,W) [float32]
        Image

    Return
    ------
    Integral image : 2D numpy array (H,W) [float32]
    */

    int H = py_image.shape(0);
    int W = py_image.shape(1);
    int L = H*W;

    const scalar_t* image = reinterpret_cast<const scalar_t*>(py_image.get_data());
    scalar_t* const integral = integral_image_impl(image, H, W);
    np::ndarray py_integral = make_ndarray(integral, H, W);
   
    return py_integral;
}

//np::ndarray integral_image_mask(np::ndarray image, np::ndarray mask){
//    /*
//    Parameters
//    ----------
//    image: 2D numpy array (H,W) [float32]
//        Image
//    mask: 2D numpy array (H,W) [bool]
//        Mask

//    Return
//    ------
//    Integral image : 2D numpy array (H,W) [float32]
//    */

//    int i,j;

//    int H = image.shape(0);
//    int W = image.shape(1);
//    int L = H*W;
//    scalar_t* integral = new scalar_t[L];
//    scalar_t* p_integral = integral;
//    scalar_t* p_image = reinterpret_cast<scalar_t*>(image.get_data());
//    bool* p_mask = reinterpret_cast<bool*>(mask.get_data());
//    scalar_t sum = 0;

//    // For the first row
//    for(j=0; j<W; ++j){
//        if(*p_mask==true){
//            sum += *(p_image++);
//        }
//        *(p_integral++) = sum;         
//    } 
//    
//    // For the rest of the rows
//    for(i=1; i<H; ++i){
//        sum = 0;
//        for(j=0; j<W; ++j){ 
//            if(*p_mask==true){
//                *(p_integral) = sum +*(p_integral-W) + *p_image;
//                sum += *p_image;
//            }else{ 
//                *(p_integral) = sum +*(p_integral-W);
//            }
//            p_integral++;
//            p_image++;
//            p_mask++;
//        }    
//    }

//    np::ndarray pyIntegral_ = np::from_data(integral, 
//                                     np::dtype::get_builtin<scalar_t>(),
//                                     p::make_tuple(L),
//                                     p::make_tuple(sizeof(scalar_t)),
//                                     p::object());   
//    return pyIntegral_.reshape(p::make_tuple(H,W));
//}

scalar_t integral_sum_impl(const scalar_t* p_integral, const int W, const int ymin, const int ymax, const int xmin, int xmax){

    const scalar_t* ptr;

    ptr = p_integral + ymin*W + xmin;
    scalar_t top_left = *ptr;

    ptr = p_integral + ymin*W + xmax;
    scalar_t top_right = *ptr;

    ptr = p_integral + ymax*W + xmax;
    scalar_t bottom_right = *ptr;

    ptr = p_integral + ymax*W + xmin;
    scalar_t bottom_left = *ptr;
  
    return bottom_right+top_left-bottom_left-top_right;
}

scalar_t integral_sum(np::ndarray py_integral, const int ymin, const int ymax, const int xmin, const int xmax){
    /// 
    /// Parameters
    /// ----------
    /// integral: 2D numpy array (H,W) [float32]
    ///     Integral image
    /// 
    /// Return
    /// ------
    /// sum : [float32]

    int W = py_integral.shape(1);
    const scalar_t* integral = reinterpret_cast<const scalar_t*>(py_integral.get_data());

    return integral_sum_impl(integral, W, ymin, ymax, xmin, xmax);
}

p::tuple compute_psi(np::ndarray Ai, np::ndarray BAi, const float lAl, const float lBxAl, const float lBl, np::ndarray py_rectangles, np::ndarray q){

    int i,j,k;
    float alpha, l1_AxAkl, lAk0l, lAk1l, lBx1_AxAkl, lBxAk0l, lBxAk1l;
    int ymin,ymax,xmin,xmax;

    int H = Ai.shape(0);
    int W = Ai.shape(1);
    int n_positions = q.shape(0);
    register int* rectangles   = reinterpret_cast<int*>(py_rectangles.get_data());
    register scalar_t* q_ = reinterpret_cast<scalar_t*>(q.get_data());
    register scalar_t* p_Ai = reinterpret_cast<scalar_t*>(Ai.get_data());
    register scalar_t* p_BAi = reinterpret_cast<scalar_t*>(BAi.get_data());
    register scalar_t* psi0 = new scalar_t[n_positions];
    register scalar_t* psi1 = new scalar_t[n_positions];
    register scalar_t* p_psi0 = psi0;
    register scalar_t* p_psi1 = psi1;

    for(k=0; k<n_positions; ++k){
        ymin = *(rectangles++);
        ymax = *(rectangles++);
        xmin = *(rectangles++);
        xmax = *(rectangles++);
        if(ymin>-1){
            // we make sure to not go outside the image
            ymin = max(ymin, 0);
            ymax = min(ymax, H-1);
            xmin = max(xmin, 0);
            xmax = min(xmax, W-1);

            l1_AxAkl = integral_sum_impl(p_Ai, W, ymin, ymax, xmin, xmax);

            alpha = -*q_/(1-*q_);
            lAk0l = lAl + alpha*l1_AxAkl;
            lAk1l = lAl +       l1_AxAkl;

            lBx1_AxAkl = integral_sum_impl(p_BAi, W, ymin, ymax, xmin, xmax);
            lBxAk0l = lBxAl + alpha*lBx1_AxAkl;
            lBxAk1l = lBxAl +       lBx1_AxAkl;

            *p_psi0 = (lBl-2*lBxAk0l)/lAk0l;
            *p_psi1 = (lBl-2*lBxAk1l)/lAk1l;
        }else{
            *p_psi0 = 1e-12;
            *p_psi1 = 1e-12;
        }
        q_++;
        p_psi0++;
        p_psi1++;
    }

    np::ndarray py_psi0 = make_ndarray(psi0, n_positions);
    np::ndarray py_psi1 = make_ndarray(psi1, n_positions);

    return p::make_tuple(py_psi0, py_psi1);
}

BOOST_PYTHON_MODULE(core) {   

    Py_Initialize();
    np::initialize();

    p::def("compute_A_", compute_A_);
    p::def("integral_image", integral_image);
    //p::def("integral_image_mask", integral_image_mask);
    p::def("integral_sum", integral_sum);
    p::def("compute_psi", compute_psi);
};
