#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import platform
import psutil
import sys
import locale
from subprocess import PIPE, Popen


class SysInfo():
    """ System and Python Information """

    def __init__(self):
        self.platform = platform.platform()
        self.system = platform.system()
        self.machine = platform.machine()
        self.release = platform.release()
        self.processor = platform.processor()
        self.cpu_count = os.cpu_count()
        self.py_implementation = platform.python_implementation()
        self.py_version = platform.python_version()

    @property
    def encoding(self):
        """ Return system preferred encoding """
        return locale.getpreferredencoding()

    @property
    def ram_free(self):
        """ return free RAM """
        return getattr(self.ram, "free")

    @property
    def ram_total(self):
        """ return total RAM """
        return getattr(self.ram, "total")

    @property
    def ram_available(self):
        """ return available RAM """
        return getattr(self.ram, "available")

    @property
    def ram_used(self):
        """ return used RAM """
        return getattr(self.ram, "used")

    @property
    def is_conda(self):
        """ Boolean for whether in a conda environment """
        return "conda" in sys.version.lower()

    @property
    def is_linux(self):
        """ Boolean for whether system is Linux """
        return self.system.lower() == "linux"

    @property
    def is_macos(self):
        """ Boolean for whether system is macOS """
        return self.system.lower() == "darwin"

    @property
    def is_windows(self):
        """ Boolean for whether system is Windows """
        return self.system.lower() == "windows"

    @property
    def is_virtual_env(self):
        """ Boolean for whether running in a virtual environment """
        if not self.is_conda:
            retval = (
                hasattr(sys, "real_prefix") or  # noqa: W504
                (
                    hasattr(sys, "base_prefix") and  # noqa: W504
                    sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = (os.path.basename(prefix) == "envs")
        return retval

    @property
    def ram(self):
        """ Return RAM stats """
        return psutil.virtual_memory()

    def format_ram(self):
        """ Format the RAM stats for human output """
        retval = list()
        for name in ("total", "available", "used", "free"):
            value = getattr(self, "ram_{}".format(name))
            value = int(value / (1024 * 1024))
            retval.append("{}: {}MB".format(name.capitalize(), value))
        return ", ".join(retval)

    @property
    def installed_pip(self):
        """ Installed pip packages """
        pip = Popen("{} -m pip freeze".format(sys.executable),
                    shell=True,
                    stdout=PIPE)
        installed = pip.communicate()[0].decode().splitlines()
        return "\n".join(installed)

    @property
    def conda_version(self):
        """ Get conda version """
        if not self.is_conda:
            return "N/A"
        conda = Popen("conda --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = conda.communicate()
        if stderr:
            return "Conda is used, but version not found"
        version = stdout.decode().splitlines()
        return "\n".join(version)

    # @property
    def full_info(self):
        """ Format system info human readable """
        retval = "\n============ System Information ============\n"
        sys_info = {
            "os_platform": self.platform,
            "os_machine": self.machine,
            "os_release": self.release,
            "py_conda_version": self.conda_version,
            "py_implementation": self.py_implementation,
            "py_version": self.py_version,
            "py_virtual_env": self.is_virtual_env,
            "sys_cores": self.cpu_count,
            "sys_processor": self.processor,
            "sys_ram": self.format_ram(),
            "encoding": self.encoding,
        }
        for key in sorted(sys_info.keys()):
            retval += ("{0: <20} {1}\n".format(key + ":", sys_info[key]))

        retval += "\n=============== Pip Packages ===============\n"
        retval += self.installed_pip

        return retval


def main():
    sysinfo = SysInfo()
    print(sysinfo.full_info())


if __name__ == "__main__":
    main()
