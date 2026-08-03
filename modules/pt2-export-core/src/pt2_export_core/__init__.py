"""Domain-agnostic core for exporting model zoos via torch.export/PT2.

No `torch` or `timm` import anywhere in this package: it works on already-
exported `ExportedProgram`/FX graphs (duck-typed on schema and
`node.meta['val']`), never constructs a model or an example input. Building
a model and its example input is inherently specific to whichever zoo (timm,
transformers, ...) is being exported, and stays with that zoo's own driver
scripts.
"""
