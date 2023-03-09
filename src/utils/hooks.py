def vitdyn_hooked_forward(model, x):
    activations = []
    hook_handles = []
    def forward_hook(module, inputs, output):
        activations.append(output)
    core = tuple(model.stem.children())[2]
    block_core = tuple(core.children())
    for block in block_core:
        handle = block.register_forward_hook(forward_hook)
        hook_handles.append(handle)
    y = model(x)

    # remove all hooks
    for handle in hook_handles:
        handle.remove()  
    return y, activations 


if __name__ == "__main__":
    import torch
    from src.models.vitdyn import vitdyn_fno_patch16_224
    fno_cfg = {
        'modes1': 16,
        'modes2': 16,
        'width' : 64, 
        'in_channels' : 3,
        'out_channels' : 3,
        'nlayers' : 4,
    }
    model = vitdyn_fno_patch16_224(pretrained=True, downstream_model=fno_cfg)
    x = torch.randn(3, 3, 224, 224)
    y, activations = vitdyn_hooked_forward(model, x)
    print(y.shape, len(activations))
    y1 = model(x)
    assert (y1 == y).all()