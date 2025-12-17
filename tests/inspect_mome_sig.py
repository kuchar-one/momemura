try:
    from qdax.core.mome import MOME
    import inspect
    import jax

    print("MOME imported successfully.")

    # Check scan_update source or signature
    print("MOME.scan_update signature:", inspect.signature(MOME.scan_update))

    # Check if jitted
    # JIT objects usually are PjitFunction or similar
    print(
        f"Is MOME.scan_update jitted? {isinstance(MOME.scan_update, (jax.jit, type(jax.jit(lambda: None))))}"
    )

    # Can we see source?
    import inspect

    try:
        src = inspect.getsource(MOME.scan_update)
        print("Source of scan_update:\n", src)
    except Exception as e:
        print("Could not get source:", e)

    try:
        src_up = inspect.getsource(MOME.update)
        print("Source of update:\n", src_up)
    except Exception as e:
        print("Could not get source of update:", e)

except ImportError:
    print("Could not import qdax.core.mome")
except Exception as e:
    print(f"Error: {e}")
