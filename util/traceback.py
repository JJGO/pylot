def setup_colored_traceback():
    try:
        import IPython.core.ultratb
    except ImportError:
        # No IPython. Use default exception printing.
        return False
    else:
        import sys
        sys.excepthook = IPython.core.ultratb.ColorTB(ostream=sys.stderr)
        return True
