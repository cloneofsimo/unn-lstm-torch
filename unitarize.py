from time import time
import torch


def unitarize(A: torch.tensor):
    """
    unitarize a matrix A
    """

    L, V = torch.linalg.eigh(torch.bmm(A.transpose(1, 2), A))
    # rint(L, V)
    return A @ V @ torch.diag_embed((L + 1e-5) ** -0.5) @ V.transpose(1, 2)


if __name__ == "__main__":

    start = time()
    D = 4
    for _ in range(1):
        A = torch.randn(4, D, D).to("cuda:0")
        U = unitarize(A)[0]
        print(U @ U.T)
        print(U)
        print(unitarize(U.unsqueeze(0)))
        print(unitarize(U.unsqueeze(0)))
        print(unitarize(U.unsqueeze(0)))
        assert torch.allclose(U @ U.T, torch.eye(D).to("cuda:0"), atol=0.1)

    print((time() - start) / 200)
