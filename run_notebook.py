"""Run the notebook in a fresh kernel using this Python interpreter."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager, KernelSpec
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parent
os.chdir(root)
for env_name, subdir in [("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "runtime"),
                         ("MPLCONFIGDIR", "matplotlib")]:
    path = root / ".jupyter" / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[env_name] = str(path)

class LocalKernelSpecManager(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                          display_name="Project Python", language="python")

nb = nbformat.read(root / "mars_rover_assignment.ipynb", as_version=4)
km = KernelManager(kernel_spec_manager=LocalKernelSpecManager())
NotebookClient(nb, km=km, timeout=300, resources={"metadata": {"path": str(root)}}).execute()
nbformat.validate(nb)
nbformat.write(nb, root / "mars_rover_assignment.ipynb")
html, _ = HTMLExporter().from_notebook_node(nb)
(root / "mars_rover_assignment.html").write_text(html, encoding="utf-8")
print("All cells executed successfully; HTML exported")
