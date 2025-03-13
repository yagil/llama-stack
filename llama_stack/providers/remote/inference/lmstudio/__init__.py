from .config import LmstudioImplConfig


async def get_adapter_impl(config: LmstudioImplConfig, _deps):
    from .lmstudio import LMStudioInferenceAdapter

    impl = LMStudioInferenceAdapter(config.url)
    await impl.initialize()
    return impl
