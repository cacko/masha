import os
import signal
from corefile import TempPath
import logging
import appnope
import typer
import psutil

from .cli import cli

appnope.nope()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

PID = os.getpid()


def shutdown(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        logging.warning(f"killing child {child}")
        child.kill()
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        logging.warning(f"killing parent {pid}")
        parent.kill()
        parent.wait(5)


def handler_stop_signals(signum, frame):
    logging.warning("Stopping app")
    TempPath.clean()
    shutdown(PID)

signal.signal(signal.SIGINT, handler_stop_signals)
signal.signal(signal.SIGTERM, handler_stop_signals)

try:
    cli()
except KeyboardInterrupt:
    raise typer.Exit()
except Exception as e:
    logging.exception(e)
    shutdown(PID)
