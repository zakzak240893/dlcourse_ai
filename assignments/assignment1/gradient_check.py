import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    if np.ndim(analytic_grad) == 1:
        analytic_grad = analytic_grad.reshape(x.shape)
    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        
        x_adjusted_up = x.copy()
        x_adjusted_down = x.copy()
        x_adjusted_up[ix] += delta
        x_adjusted_down[ix] -= delta
        #print(ix, x_adjusted_up, x_adjusted_down, f(x_adjusted_up)[0],f(x_adjusted_down)[0],  sep = '\r\n'*2)
        
        numeric_grad_at_ix = (f(x_adjusted_up)[0]  - f(x_adjusted_down)[0])/2/delta
        #print(numeric_grad_at_ix)
        #print(f'{ix}({f(x_adjusted_up)[0]}  - {f(x_adjusted_down)[0]})/(2*delta) = {numeric_grad_at_ix}|{analytic_grad_at_ix}|{numeric_grad_at_ix/analytic_grad_at_ix}')

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True