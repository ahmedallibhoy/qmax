import subprocess


def on_config(config, **kwargs):
    """Point the [source] links in the API docs at the commit being built.

    Consumed by `docs/_templates/python/material/{class,function}.html.jinja`
    as `config.extra.source_url`.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    ref = head.stdout.strip() if head.returncode == 0 else "main"
    options = config.plugins["mkdocstrings"].config.handlers["python"]["options"]
    options.setdefault("extra", {})["source_url"] = f"{config.repo_url}/blob/{ref}"
    return config
