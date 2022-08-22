import cuqi

def test_density_variable_name_detection():
    """Test that the density variable name is detected correctly at different levels of the python stack. """

    # Test that the density variable name is detected correctly at current level.
    x = cuqi.distribution.GaussianCov()
    assert x.name == 'x'

    # Test that variable name is detected correctly 1 level deep.
    def inner_name():
        y = cuqi.distribution.GaussianCov()
        assert y.name == 'y'
    inner_name()

    # Test variable name is detected correctly at n levels deep.
    class recursive_name:
        def __init__(self, max_recursion=10):
            self.max_recursion = max_recursion
        def __call__(self, current_recursion=0):
            if current_recursion == self.max_recursion:
                z = cuqi.distribution.GaussianCov()
                assert z.name == 'z'
            else:
                self(current_recursion + 1)
    recursive_name()()

    
