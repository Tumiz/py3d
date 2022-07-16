import pathlib
import nbformat
import nbclient
import nbconvert
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--destination","-d",default="docs")
args = parser.parse_args()

path_destination = pathlib.Path(args.destination)
path_doc = pathlib.Path(__file__).parent/"py3d/doc"

for f in path_doc.glob("*"):
    if f.suffix == ".ipynb":
        print(f.name)
        nb=nbformat.read(f, as_version=4)
        nbclient.client.NotebookClient(nb).execute()
        i = 0
        for cell in nb.cells:
            if "assert" in cell.source:
                del nb.cells[i]
            i += 1
        body, resources = nbconvert.HTMLExporter().from_notebook_node(nb)
        open(path_destination/(f.stem+".html"),"w").write(body)