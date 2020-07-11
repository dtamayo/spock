import subprocess
__version__ = '1.0.3'
__githash__ = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
