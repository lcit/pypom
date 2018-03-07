#include <boost/python/numpy.hpp>
#include <iostream>

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

using scalar_t = double; // double is faster than float

np::ndarray compute_A_(p::tuple view_shape, np::ndarray rectangles, np::ndarray q){

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

    int i,j,k;
    float proba_absence;
    int ymin,ymax,xmin,xmax;

    int H = p::extract<int>(view_shape[0]);
    int W = p::extract<int>(view_shape[1]);
    int L = H*W;
    register int* r_   = reinterpret_cast<int*>(rectangles.get_data());
    register scalar_t* q_ = reinterpret_cast<scalar_t*>(q.get_data());
    register scalar_t* A_ = new scalar_t[L];
    for(i=0; i<L; ++i){
        A_[i] = 1.0;   
    }    

    for(k=0; k<q.shape(0); ++k){
        ymin = *(r_++);
        ymax = *(r_++);
        xmin = *(r_++);
        xmax = *(r_++);
        if(ymin>-1){

            //cout << ymin << "," << ymax << "," << xmin << "," << xmax << endl;
            //cout << *q_ << endl;

            proba_absence = 1.0 - *q_;
            //cout <<  proba_absence << "---" << *q_ << endl;
                     
            for(i=ymin; i<ymax; ++i){
                register scalar_t* ptr = A_;
                ptr += i*W+xmin;
                for(j=xmin; j<xmax; ++j){ 
                    *ptr *= proba_absence;
                    ptr++;
                }   
            }  
        } 
        q_++;   
    }
    
    np::ndarray pyA_ = np::from_data(A_, 
                                     np::dtype::get_builtin<scalar_t>(),
                                     p::make_tuple(L),
                                     p::make_tuple(sizeof(scalar_t)),
                                     p::object());    
    return pyA_.reshape(p::make_tuple(H,W));
}

np::ndarray integral_image(np::ndarray image){
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

    int H = image.shape(0);
    int W = image.shape(1);
    int L = H*W;
    scalar_t* integral = new scalar_t[L];
    scalar_t* p_integral = integral;
    scalar_t* p_image = reinterpret_cast<scalar_t*>(image.get_data());
    float sum = 0;

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

    np::ndarray pyIntegral_ = np::from_data(integral, 
                                     np::dtype::get_builtin<scalar_t>(),
                                     p::make_tuple(L),
                                     p::make_tuple(sizeof(scalar_t)),
                                     p::object());   
    return pyIntegral_.reshape(p::make_tuple(H,W));
}

np::ndarray integral_image_mask(np::ndarray image, np::ndarray mask){
    /*
    Parameters
    ----------
    image: 2D numpy array (H,W) [float32]
        Image
    mask: 2D numpy array (H,W) [bool]
        Mask

    Return
    ------
    Integral image : 2D numpy array (H,W) [float32]
    */

    int i,j;

    int H = image.shape(0);
    int W = image.shape(1);
    int L = H*W;
    scalar_t* integral = new scalar_t[L];
    scalar_t* p_integral = integral;
    scalar_t* p_image = reinterpret_cast<scalar_t*>(image.get_data());
    bool* p_mask = reinterpret_cast<bool*>(mask.get_data());
    scalar_t sum = 0;

    // For the first row
    for(j=0; j<W; ++j){
        if(*p_mask==true){
            sum += *(p_image++);
        }
        *(p_integral++) = sum;         
    } 
    
    // For the rest of the rows
    for(i=1; i<H; ++i){
        sum = 0;
        for(j=0; j<W; ++j){ 
            if(*p_mask==true){
                *(p_integral) = sum +*(p_integral-W) + *p_image;
                sum += *p_image;
            }else{ 
                *(p_integral) = sum +*(p_integral-W);
            }
            p_integral++;
            p_image++;
            p_mask++;
        }    
    }

    np::ndarray pyIntegral_ = np::from_data(integral, 
                                     np::dtype::get_builtin<scalar_t>(),
                                     p::make_tuple(L),
                                     p::make_tuple(sizeof(scalar_t)),
                                     p::object());   
    return pyIntegral_.reshape(p::make_tuple(H,W));
}

scalar_t integral_sum_impl(scalar_t* p_integral, int W, int ymin, int ymax, int xmin, int xmax){

    scalar_t* ptr;

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

scalar_t integral_sum(np::ndarray integral, int ymin, int ymax, int xmin, int xmax){
    /// 
    /// Parameters
    /// ----------
    /// integral: 2D numpy array (H,W) [float32]
    ///     Integral image
    /// 
    /// Return
    /// ------
    /// sum : [float32]

    int i,j;

    int H = integral.shape(0);
    int W = integral.shape(1);
    int L = H*W;
    scalar_t* p_integral = reinterpret_cast<scalar_t*>(integral.get_data());
    scalar_t* ptr = p_integral;

    ptr = p_integral + ymin*W + xmin;
    scalar_t top_left = *ptr;

    ptr = p_integral + ymin*W + xmax;
    scalar_t top_right = *ptr;

    ptr = p_integral + ymax*W + xmax;
    scalar_t bottom_right = *ptr;

    ptr = p_integral + ymax*W + xmin;
    scalar_t bottom_left = *ptr;

    //cout << bottom_right << endl;
  
    return bottom_right+top_left-bottom_left-top_right;
}

p::tuple compute_psi(np::ndarray Ai, np::ndarray BAi, float lAl, float lBxAl, float lBl, np::ndarray rectangles, np::ndarray q){

    int i,j,k;
    float alpha, l1_AxAkl, lAk0l, lAk1l, lBx1_AxAkl, lBxAk0l, lBxAk1l;
    int ymin,ymax,xmin,xmax;

    int W = Ai.shape(1);
    int n_positions = q.shape(0);
    register int* r_   = reinterpret_cast<int*>(rectangles.get_data());
    register scalar_t* q_ = reinterpret_cast<scalar_t*>(q.get_data());
    register scalar_t* p_Ai = reinterpret_cast<scalar_t*>(Ai.get_data());
    register scalar_t* p_BAi = reinterpret_cast<scalar_t*>(BAi.get_data());
    register scalar_t* psi0 = new scalar_t[n_positions];
    register scalar_t* psi1 = new scalar_t[n_positions];
    register scalar_t* p_psi0 = psi0;
    register scalar_t* p_psi1 = psi1;

    for(k=0; k<n_positions; ++k){
        ymin = *(r_++);
        ymax = *(r_++);
        xmin = *(r_++);
        xmax = *(r_++);
        if(ymin>-1){

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
            *p_psi0 = 1e-6;
            *p_psi1 = 1e-6;
        }
        q_++;
        p_psi0++;
        p_psi1++;
    }

    np::ndarray pyPsi0 = np::from_data(psi0, 
                                       np::dtype::get_builtin<scalar_t>(),
                                       p::make_tuple(n_positions),
                                       p::make_tuple(sizeof(scalar_t)),
                                       p::object()); 
    np::ndarray pyPsi1 = np::from_data(psi1, 
                                       np::dtype::get_builtin<scalar_t>(),
                                       p::make_tuple(n_positions),
                                       p::make_tuple(sizeof(scalar_t)),
                                       p::object());
    return p::make_tuple(pyPsi0, pyPsi1);
}

BOOST_PYTHON_MODULE(pom_core) {   

    Py_Initialize();
    np::initialize();

    p::def("compute_A_", compute_A_);
    p::def("integral_image", integral_image);
    p::def("integral_image_mask", integral_image_mask);
    p::def("integral_sum", integral_sum);
    p::def("compute_psi", compute_psi);
};
