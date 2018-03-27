/*  ==========================================================================================
    Author: Leonardo Citraro
    Institution: Computer Vision Laboratory - École polytechnique fédérale de Lausanne
    Description: POM core functionalities
    
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
#ifndef CORE_HPP
#define CORE_HPP

#include <iostream>
#include <cmath> 

namespace core {

using scalar_t = float;

/** 
  *   @brief  Clamping a scalar value.   
  */  
template<typename T>
inline T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}
 
/** 
  *   @brief  Sum elements of an array.   
  *  
  *   @param  array pointer to a 1D array 
  *   @param  L length of the array
  *   @return sum
  */ 
scalar_t sum(const scalar_t* array, const int L){
    int i;
    scalar_t sum = 0;
    for(i=0; i<L; ++i){
        sum += *(array++);   
    }   
    return sum;
}

/** 
  *   @brief  Sum elements of an array with mask.   
  *  
  *   @param  array pointer to a 1D array 
  *   @param  mask pointer to a 1D array
  *   @param  L length of the arrays
  *   @return sum
  */ 
scalar_t sum_mask(const scalar_t* array, const scalar_t* mask, const int L){
    int i;
    scalar_t sum = 0;
    for(i=0; i<L; ++i){
        if(*mask>0){
            sum += *array; 
        } 
        mask++;
        array++; 
    }  
    return sum;
}
 
/** 
  *   @brief  Column-wise sum (sum of rows).   
  *  
  *   @param  array pointer to a 2D array 
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns) 
  *   @return array of size W
  */ 
scalar_t* const sum_rows(scalar_t const** array, const int H, const int W){
    int h,w;
    scalar_t* const sum = new scalar_t[W]();
    for(w=0; w<W; ++w){        
        for(h=0; h<H; ++h){
            sum[w] += array[h][w];
        }  
    }   
    return sum;        
}

/** 
  *   @brief  Row-wise sum (sum of columns).   
  *  
  *   @param  array pointer to a 2D array 
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns) 
  *   @return array of size H
  */
scalar_t* const sum_cols(const scalar_t** array, const int H, const int W){
    int h;
    scalar_t* const _sum = new scalar_t[H];
    for(h=0; h<H; ++h){
        _sum[h] = sum(array[h], W);   
    }   
    return _sum;
}

/** 
  *   @brief  Part of equation (31) of the paper.   
  *  
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns) 
  *   @param  n_positions number of rectangles/positions
  *   @param  rectangles the rectangles for a specific view
  *           in the format [ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...]
  *   @return q probability of presence
  */
scalar_t* const compute_A_(const int H, const int W, const int n_positions, 
                           const int* rectangles, const scalar_t* q){
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
        
        // case when the rectangle is visible
        if(ymin>-1){
            // we make sure to not go outside the image
            ymin = clip(ymin, 0, H);
            ymax = clip(ymax, 0, H);
            xmin = clip(xmin, 0, W);
            xmax = clip(xmax, 0, W);

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

/** 
  *   @brief  Computes the integral image/array.   
  *  
  *   @param  image the image as 1D array
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns) 
  *   @return the integral image as 1D array, same size as image
  */
scalar_t* const integral_image(const scalar_t* image, const int H, const int W){
    int h,w;
    int L = H*W;
    scalar_t sum = 0;
    scalar_t* const integral = new scalar_t[L];
    scalar_t* p_integral = integral;    

    // For the first row
    for(w=0; w<W; ++w){
        sum += *(image++);
        *(p_integral++) = sum;         
    } 
    
    // For the rest of the rows
    for(h=1; h<H; ++h){
        sum = 0;
        for(w=0; w<W; ++w){ 
            *p_integral = sum +*(p_integral-W) + *image; 
            sum += *image;
            p_integral++;
            image++;
        }    
    }
    return integral;
}

/** 
  *   @brief  Computes the integral image/array with mask.   
  *  
  *   @param  image the image as 1D array
  *   @param  mask the mask as 1D array [0,1]
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns) 
  *   @return the integral image as 1D array, same size as image
  */
scalar_t* const integral_image_mask(const scalar_t* image, const scalar_t* mask, 
                                    const int H, const int W){
    int h,w;
    int L = H*W;
    scalar_t sum = 0;
    scalar_t* const integral = new scalar_t[L];
    scalar_t* p_integral = integral;    

    // For the first row
    for(w=0; w<W; ++w){
        sum += *(image++) * *(mask++);
        *(p_integral++) = sum;         
    } 
    
    // For the rest of the rows
    for(h=1; h<H; ++h){
        sum = 0;
        for(w=0; w<W; ++w){ 
            *(p_integral) = sum +*(p_integral-W) + *(image) * *(mask); 
            sum += *(image) * *(mask);
            p_integral++;
            image++;
            mask++;
        }    
    }
    return integral;
}

/** 
  *   @brief  Computes the sum of a rectangular region 
  *           in the image using the integral image.   
  *  
  *   @param  integral the integral image as 1D array   
  *   @param  W second dimension of image size (number of columns) 
  *   @param  (xmin, ymin) is upper left corner of the box/rectangle
  *   @param  (xmax, ymax) is bottom right corner of the box/rectangle      
  *   @return the sum of elements inside the box
  */
scalar_t integral_sum(const scalar_t* integral, const int W, 
                      const int ymin, const int ymax, const int xmin, const int xmax){
    const scalar_t* ptr;

    ptr = integral + ymin*W + xmin;
    scalar_t top_left = *ptr;

    ptr = integral + ymin*W + xmax;
    scalar_t top_right = *ptr;

    ptr = integral + ymax*W + xmax;
    scalar_t bottom_right = *ptr;

    ptr = integral + ymax*W + xmin;
    scalar_t bottom_left = *ptr;
  
    return bottom_right+top_left-bottom_left-top_right;
}

/** 
  *   @brief  Computes part of equations (35) for one view/camera passing from  
  *           equations (31),(32),(33) and (34) of the paper.    
  *  
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns) 
  *   @param  Ai integral image (1-A) as 1D array
  *   @param  BAi integral image Bx(1-A) as 1D array  
  *   @param  lAl 
  *   @param  lBxAl  
  *   @param  lBl    
  *   @param  rectangles the rectangles for a specific view/camera
  *           in the format [ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...]  
  *   @param q probability of presence
  *   @param  n_positions number of rectangles/positions
  *   @return psi(B, A1) - psi(B, A0) 1D array of size n_positions
  */
scalar_t* const compute_psi_diff(const int H, const int W, const scalar_t* Ai, const scalar_t* BAi, 
                                 const scalar_t lAl, const scalar_t lBxAl, const scalar_t lBl, 
                                 const int* rectangles, const scalar_t* q, const int n_positions){

    int i,j,k;
    scalar_t alpha, l1_AxAkl, lAk0l, lAk1l, lBx1_AxAkl, lBxAk0l, lBxAk1l;
    int ymin,ymax,xmin,xmax;

    scalar_t* const psi_diff = new scalar_t[n_positions];

    for(k=0; k<n_positions; ++k){
        ymin = *(rectangles++);
        ymax = *(rectangles++);
        xmin = *(rectangles++);
        xmax = *(rectangles++);
        
        if(ymin<0){
            // the case when the rectangle is not visible
            psi_diff[k] = 0;       
        }else{
            // we make sure to not go outside the image
            ymin = clip(ymin, 0, H);
            ymax = clip(ymax, 0, H);
            xmin = clip(xmin, 0, W);
            xmax = clip(xmax, 0, W);

            alpha = -q[k]/(1-q[k]);
            
            l1_AxAkl = integral_sum(Ai, W, ymin, ymax, xmin, xmax);            
            lAk0l = lAl + alpha*l1_AxAkl;
            lAk1l = lAl +       l1_AxAkl;

            lBx1_AxAkl = integral_sum(BAi, W, ymin, ymax, xmin, xmax);
            lBxAk0l = lBxAl + alpha*lBx1_AxAkl;
            lBxAk1l = lBxAl +       lBx1_AxAkl;

            psi_diff[k] = (lBl-2*lBxAk1l)/lAk1l - (lBl-2*lBxAk0l)/lAk0l; // psi(B, A1) - psi(B, A0)
        }      
    }
    return psi_diff;
}

// This function is never used because my implementation of the sum in C++ is 
// slower then numpy.sum(). For this reason the final algorithm is mostly written in Python
// but with some C++ code. Have a look at the class Solver of the file pom.py
/** 
  *   @brief  The full detection algorithm.    
  *  
  *   @param  B the background substraction images as 2D array
  *   @param  rectangles the rectangles for a specific view/camera
  *           in the format [ymin1,ymax1,xmin1,xmax1, ymin2,ymax2,xmin2,xmax2, ...]  
  *   @param @return  q probability of presence. This is modified inplace 
  *   @param  n_cams number of cameras/views
  *   @param  n_positions number of rectangles/positions
  *   @param  H first dimension of image size (number of rows)   
  *   @param  W second dimension of image size (number of columns)   
  *   @param  prior prior prior probability of presence
  *   @param  sigma quality of the background segmentation images
  *   @param  step ~learning rate for the optimization
  *   @param  max_iter maximum numer of iteration for the optimization
  *   @param  tol tollerance for the optimization, if reached the functions returns                
  *   @return the number of iterations performed. 
  */
int solve(const scalar_t** B, const int** rectangles, scalar_t* q, 
          const int n_cams, const int n_positions, const int H, const int W, 
          const scalar_t prior, const scalar_t sigma, const scalar_t step, 
          const int max_iter, const scalar_t tol){
    int i,j,k,c;
    int L = H*W;
    int n_stab = 0;
    scalar_t exp_lambda = (1-prior)/prior; 
    scalar_t qk_new, sigma_sum_psi, mean_diff;
    
    scalar_t** const A = new scalar_t*[n_cams];
    for(c=0; c<n_cams; ++c){
        A[c] = new scalar_t[L];
    }

    scalar_t** const Ai = new scalar_t*[n_cams]; // integral image (1-A)
    scalar_t** const BAi = new scalar_t*[n_cams]; // integral image Bx(1-A)
    scalar_t** const psi_diff = new scalar_t*[n_cams]; // psi(B,A1)-psi(B,A0)

    scalar_t* A_;
    scalar_t* lAl = new scalar_t[n_cams];
    scalar_t* lBxAl = new scalar_t[n_cams];
    scalar_t* lBl = new scalar_t[n_cams]; 
    scalar_t* sum_psi;

    for(i=0; i<max_iter; ++i){
        for(c=0; c<n_cams; ++c){
            
            A_ = compute_A_(H, W, n_positions, rectangles[c], q);
            for(j=0; j<L; ++j){
                A[c][j] = 1-A_[j]; // 1-xk(1-qkAk)
            }
            
            Ai[c] = integral_image(A_,H,W); // integral image (1-A)
            BAi[c] = integral_image_mask(A_,B[c],H,W); // integral image Bx(1-A)

            lAl[c] = sum(A[c], L); // |A|
            lBxAl[c] = sum_mask(A[c], B[c], L); // |BxA|
            lBl[c] = sum(B[c], L); // |B|

            psi_diff[c] = compute_psi_diff(H, W, Ai[c], BAi[c], lAl[c], lBxAl[c], 
                                             lBl[c], rectangles[c], q, n_positions);
                       
        }        
        sum_psi = sum_rows((scalar_t const**)psi_diff, n_cams, n_positions);

        // q update
        mean_diff = 0;
        for(k=0; k<n_positions; ++k){
            sigma_sum_psi = std::min(sum_psi[k]/sigma, (scalar_t)30); // clamping to avoid overflow on the exp(.)
            qk_new = q[k]*step + (1-step)/(1+exp_lambda*std::exp(sigma_sum_psi));
            mean_diff += std::abs(qk_new - q[k]);
            q[k] = qk_new;
        }          

        // early stopping
        if(mean_diff/n_positions < tol){
            if(++n_stab > 5){ 
                return i;
            }
        }else{
            n_stab = 0;
        }        
    }
    return i;
}   

};

#endif      
