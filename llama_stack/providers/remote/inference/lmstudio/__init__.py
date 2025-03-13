from .config import LmstudioImplConfig


async def get_adapter_impl(config: LmstudioImplConfig, _deps):
    from .lmstudio import LmstudioInferenceAdapter

    impl = LmstudioInferenceAdapter(config.url)
    await impl.initialize()
    return impl
