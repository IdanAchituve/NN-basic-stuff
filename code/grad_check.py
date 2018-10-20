import numpy as np

STUDENT={'name': 'Idan Achituve',
         'ID': '300083029'}

def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradientscode
    - x is the point (numpy array) to check the gradient at (the parameters in the NN)
    """ 
    fx, grad = f(x) # Evaluate function value at original point. i.e., the gradient of the cost function w.r.t all parameters
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # allow iterating over arrays
    while not it.finished:
        ix = it.multi_index # give the index (row and col) in the form of tuple

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.
        ### YOUR CODE HERE:
        x_plus = np.copy(x) # create a copy with different reference
        x_minus = np.copy(x) # create a copy with different reference

        x_plus[ix] += h
        x_minus[ix] -= h

        cost_plus = f(x_plus)[0]
        cost_minus = f(x_minus)[0]

        numeric_gradient = (cost_plus-cost_minus)/(2*h)
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient)
            return
    
        it.iternext() # Step to next index
    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2) # an anonymous function which return a tuple of 2 elements: scalar and a tupple (array)
                                             # ((sum of all elements is x)^2 , multiplication of all elements in x by 2)
                                             # the shape of the 2nd element has the same shape of x

    print "Running sanity checks..."
    gradient_check(quad, np.array(123.456))      # scalar test
    gradient_check(quad, np.random.randn(3,))    # 1-D test
    gradient_check(quad, np.random.randn(4,5))   # 2-D test
    print ""

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()
