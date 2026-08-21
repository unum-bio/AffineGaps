"""Compiles the Mojo extension into the wheel, when a Mojo toolchain is present.

The gate is whether the Mojo package imports, never a platform list: Modular ships wheels for
Linux x86-64, Linux aarch64 and macOS arm64 today and may add more, and anywhere it does not ship
one this hook simply produces the pure-Python wheel.

A freshly built extension records its runtime dependencies by absolute path into the build
environment, which the installer deletes afterwards, so the libraries are copied in beside it and
the loader is repointed at the wheel's own directory. Without that the wheel installs and then
fails to import.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.path.insert(0, str(pathlib.Path(__file__).parent / "scripts"))


class MojoExtensionHook(BuildHookInterface):
    PLUGIN_NAME = "mojo"

    def initialize(self, version, build_data):
        try:
            from mojo.run import subprocess_run_mojo
        except ImportError:
            return  # No toolchain for this platform, so ship the reference alone.

        from build_extension import library_source

        source = pathlib.Path(self.root) / "affinegaps.mojo"
        staging = pathlib.Path(tempfile.mkdtemp(prefix="affinegaps-"))
        stripped = staging / "affinegaps.mojo"
        stripped.write_text(library_source(source.read_text()))

        extension = staging / self._extension_name()
        outcome = subprocess_run_mojo(["build", str(stripped), "--emit", "shared-lib", "-o", str(extension)])
        if outcome.returncode != 0 or not extension.exists():
            return

        bundled = [extension, *self._bundle_runtime(extension, staging)]
        for library in bundled:
            self._repoint(library)
            build_data["force_include"][str(library)] = library.name
        # The extension links no libpython, so one artifact serves every Python version.
        build_data["tag"] = f"py3-none-{self._platform_tag()}"
        build_data["pure_python"] = False

    @staticmethod
    def _platform_tag() -> str:
        """The wheel's platform tag, with the punctuation wheels do not allow."""
        from packaging.tags import sys_tags

        return next(tag.platform for tag in sys_tags())

    @staticmethod
    def _extension_name() -> str:
        return "affinegaps_mojo.dylib" if sys.platform == "darwin" else "affinegaps_mojo.so"

    @staticmethod
    def _bundle_runtime(extension: pathlib.Path, staging: pathlib.Path) -> list:
        """Copies in the Modular runtime libraries the extension resolves against.

        They are discovered by asking the loader rather than hardcoded, so a release that renames
        them does not silently produce a wheel that cannot import.
        """
        copied = []
        if sys.platform == "darwin":
            listing = subprocess.run(["otool", "-L", str(extension)], capture_output=True, text=True)
            candidates = [line.split()[0] for line in listing.stdout.splitlines()[1:] if line.strip()]
        else:
            listing = subprocess.run(["ldd", str(extension)], capture_output=True, text=True)
            candidates = [
                part.split(" (")[0].strip()
                for line in listing.stdout.splitlines()
                if "=>" in line
                for part in [line.split("=>")[1]]
            ]
        for candidate in candidates:
            path = pathlib.Path(candidate)
            if path.is_file() and "modular" in path.parts:
                copied.append(pathlib.Path(shutil.copy2(path, staging / path.name)))
        return copied

    @staticmethod
    def _repoint(library: pathlib.Path) -> None:
        """Points the library at whatever directory it ends up installed in."""
        if sys.platform == "darwin":
            subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", str(library)], check=False)
        else:
            subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(library)], check=False)
