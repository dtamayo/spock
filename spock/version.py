__version__ = '1.0.4'
try:
    import subprocess
    __githash__ = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
except:
    pass
