import torch

# cpu -- gpu
def ExchangeDataBetweenCpuGpu():
    torch.cuda.set_device(2)
    if torch.cuda.is_available():
        print("is available")
        device = torch.device("cuda")
        x = torch.randn(5, 3)
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print("z = ", z)
        print(z.to("cpu"), torch.double)

# autograd
def AutoGrad():
    """
    requires_grad
    grad_fn
    grad
    item()
    requires_grad_()
    backward()
    torch.no_grad()
    detach()
    """
    # requires_grad grad_fn
    x = torch.ones((2, 2), requires_grad=True)
    y = x + 2
    print("y = ", y)
    print("y.grad_fn = ", y.grad_fn)

    z = y * y * 3
    out = z.mean()
    print(z, out, out.item())

    # requires_grad_
    a = torch.ones((4, 4), dtype=torch.float)
    a = ((a * 3) / (a - 0.2))
    print("a.requires_grad = ", a.requires_grad)
    a.requires_grad_(True)
    print("a.requires_grad = ", a.requires_grad)
    b = (a * a).sum()
    print("b.grad_fn = ", b.grad_fn)

    ## gradient
    out.backward()
    print("x.grad = ", x.grad)

    ## vector-Jacbian product
    x = torch.randn(3, requires_grad=True)
    print("x = ", x)

    y = x ** 2
    y.backward(torch.ones(3))
    print("x.grad = ", x.grad)

# ## torch.nn
# def NN():
#     print("torch")
#     torch.nn.Module?


if __name__ == "__main__":
    # ExchangeDataBetweenCpuGpu();
    AutoGrad();
    # NN()


