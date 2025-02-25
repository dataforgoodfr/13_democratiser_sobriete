Pour rendre l'environnement uv disponible depuis un jupyter notebook :

```
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
```
