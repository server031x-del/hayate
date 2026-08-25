from __future__ import annotations

from hayate.runtime import Stage, StageRuntime


class Resource:
    def __init__(self):
        self.unloaded = False

    def unload(self):
        self.unloaded = True


def test_stage_runtime_releases_resources_in_order():
    runtime = StageRuntime()
    resource = Resource()
    with runtime.stage(Stage.PROMPT_ENCODING) as stage:
        stage.register(Stage.PROMPT_ENCODING, resource)
    assert resource.unloaded
    assert runtime.events[0].released_resources == 1
    with runtime.stage(Stage.DENOISING):
        pass

