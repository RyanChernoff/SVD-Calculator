using LinearAlgebra
using RowEchelon

eye(n) = (1.0 + 0im)*Matrix(I, n, n)

function getError(A, n)
    e = 0
    for i=1:n
        for j=1:n
            if i != j
                e += abs(A[i,j])^2
            end
        end
        end
    return e
end

function qr_algorithm(A, n)
    X = eye(n)
    i = 0
    while i < 10000 && getError(A, n) > 1e-16
        Q, R = qr(A)
        A = R*Q
        X = X*Q
        i+=1
    end
    return X, A
end

function finishU(A, U, m)
    N = nullspace(A’)
    n = size(N, 2) - 1
    if n < 0
        return U
    end
    U[:, (m - n):m] = N
    return U
end

function svd_qr(A)
    m = size(A, 1)
    n = size(A, 2)
    if m > n
        U, S , V = svd_qr(A’)
        return V, S’, U
    end
    V, L = qr_algorithm(A’*A,n)
    U = (0 + 0im)zeros(m, m)
    S = eye(m, n)
    for i=1:m
        S[i,i] = sqrt(round(L[i,i]; digits = 5))
        if S[i,i] != 0
            U[:, i] = (A*V[:,i])/S[i,i]
        end
    end
    U = finishU(A, U, m)
    return U, S, V
end    