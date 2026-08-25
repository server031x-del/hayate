"""Local, ComfyUI-independent HAYATE WebUI."""


def create_app(*args, **kwargs):
    from hayate.webui.server import create_app as factory

    return factory(*args, **kwargs)


def run_webui(*args, **kwargs):
    from hayate.webui.server import run_webui as runner

    return runner(*args, **kwargs)


__all__ = ["create_app", "run_webui"]
