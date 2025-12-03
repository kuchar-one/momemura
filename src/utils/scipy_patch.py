
import scipy.integrate

# Patch scipy.integrate.simps which was removed in SciPy 1.14.0
if not hasattr(scipy.integrate, 'simps'):
    if hasattr(scipy.integrate, 'simpson'):
        scipy.integrate.simps = scipy.integrate.simpson
    else:
        # Fallback or warning if simpson is also missing (unlikely in recent scipy)
        pass
