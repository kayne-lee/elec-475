def build_model(name, pretrained=True):
    """'student' = the provided StudentNet, 'custom' = our LiteSegNet (Step 3)."""
    if name == 'student':
        from .student import StudentNet
        return StudentNet(pretrained=pretrained)
    if name == 'custom':
        from .custom import LiteSegNet
        return LiteSegNet(pretrained=pretrained)
    raise ValueError(f'unknown model {name}')
