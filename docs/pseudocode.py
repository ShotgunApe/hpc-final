def gj_torch(A, b):
    A = torch.tensor(A, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    n = len(A)

    b = torch.reshape(b, (n, 1))
    Ab = torch.cat((A, b), dim = 1)

    for k in range(n):
        row_to_top_torch(Ab, k, n)

        # Division
        pivot = Ab[k, k].clone()
        Ab[k] = torch.div(Ab[k], pivot)

        # Elimination
        for i in range(n):
            if i == k or Ab[i,k] == 0:
                continue
            factor = Ab[i,k].clone()
            Ab[i, k:] = torch.sub(Ab[i, k:], torch.mul(Ab[k, k:], factor))

    return Ab[:, n:], Ab[:, :n]
