import numpy as np
import time
from colorama import Fore, Style, init
from engine import Value
init(autoreset=True)

def test_addition():
    a = Value([1, 2, 3])
    b = Value([4, 5, 6])
    c = a + b
    assert np.array_equal(c.data, np.array([5, 7, 9])), "addition test failed"

def test_gradient():
    a = Value([1., 2., 3.])
    b = Value([4., 5., 6.])
    c = a + b
    c.grad = np.ones_like(c.data)  # grad
    c._grad_fn()  # run backprop
    assert np.array_equal(a.grad, np.ones_like(a.data)), "grad test failed"
    assert np.array_equal(b.grad, np.ones_like(b.data)), "grad test failed"

def run_tests():
    tests = [test_addition, test_gradient]
    for test in tests:
        print(Fore.YELLOW + "Running: " + test.__name__)
        test()
        print(Fore.GREEN + "Passed: " + test.__name__)
        print(Fore.CYAN + "."*30)
        time.sleep(0.3)

if __name__ == "__main__":
    run_tests()
