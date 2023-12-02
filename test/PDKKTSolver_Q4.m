function [d_x, d_lambda, d_nu] = PDKKTSolver_Q4(...
    x, lambda, nu, c, A, b, t)
% PDKKTSolver_Q4

% Jingyu Liu, December 2, 2023.

arguments (Input)
    x (:, 1) double;
    lambda (:, 1) double;
    nu (:, 1) double;
    c (:, 1) double;
    A (:, :) double;
    b (:, 1) double;
    t (:, 1) double;
end

arguments (Output)
    d_x (:, 1) double;
    d_lambda (:, 1) double;
    d_nu (:, 1) double;
end

[p, n] = size(A);
m = 2 * n;
one_vec = ones(m, 1);

G_x = [-x; x - 1];

% psi = diagonal elements of Hinv.
deno = lambda(1 : n) .* (1 - x) + lambda((n + 1) : end) .* x;
nume = (1 - x) .* x;
psi = nume ./ deno;
Hinv_c = psi .* c;
Hinv_second_term = (2 * x - 1) ./ deno;
Hinv_third_term = psi .* (A' * nu);
Hinv_g = Hinv_c + Hinv_second_term / t + Hinv_third_term;

d_nu = (A * (psi .* A')) \ (A * (x - Hinv_g) - b);
d_x = -Hinv_g - psi .* A' * d_nu;
r_cent = -lambda .* G_x - one_vec / t;
d_lambda = G_x .\ (-lambda .* [-d_x; d_x] + r_cent);

end