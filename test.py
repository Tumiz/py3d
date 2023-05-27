import pathlib
import nbformat
import nbclient
import nbconvert
import argparse
import os

root = pathlib.Path(__file__).absolute().parent
print(root)
parser = argparse.ArgumentParser()
parser.add_argument("--destination", "-d", default="docs")
parser.add_argument("--notebooks", "-n",
                    default=root/"py3d/doc")
args = parser.parse_args()

path_destination = pathlib.Path(args.destination).absolute()
path_doc = pathlib.Path(args.notebooks).absolute()
if path_doc.is_dir():
    docs = path_doc.glob("*")
else:
    docs = [path_doc]

for f in docs:
    if f.suffix == ".ipynb":
        print(f.name)
        nb = nbformat.read(f, as_version=4)
        os.chdir(f.parent)
        nbclient.client.NotebookClient(nb).execute()
        for i, cell in enumerate(nb.cells):
            if "assert" in cell.source:
                del nb.cells[i]
        body, _ = nbconvert.HTMLExporter().from_notebook_node(nb)
        body = body.replace("<title>Notebook</title>",
                            "<title>Scenario {}</title>".format(f.stem))
        open(path_destination/(f.stem+".html"), "w").write(body)
        if f.name == "index.ipynb":
            for i, cell in enumerate(nb.cells):
                if hasattr(cell, "outputs"):
                    setattr(cell, "outputs", [])
                if "<script>" in cell.source:
                    del nb.cells[i]
            body, _ = nbconvert.MarkdownExporter().from_notebook_node(nb)
            open(root/"py3d/README.md", "w").write(body)
            open(root/"README.md", "w").write(body)
