import pathlib
import nbformat
import nbclient
import nbconvert
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--destination", "-d", default="docs")
parser.add_argument("--notebooks", "-n",
                    default=pathlib.Path(__file__).parent/"py3d/doc")
args = parser.parse_args()

path_destination = pathlib.Path(args.destination)
path_doc = pathlib.Path(args.notebooks)
if path_doc.is_dir():
    docs = path_doc.glob("*")
else:
    docs = [path_doc]

for f in docs:
    if f.suffix == ".ipynb":
        print(f.name)
        nb = nbformat.read(f, as_version=4)
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
                if "<script>" in cell.source:
                    del nb.cells[i]
                    break
            body, _ = nbconvert.MarkdownExporter().from_notebook_node(nb)
            open("py3d/README.md", "w").write(body)
