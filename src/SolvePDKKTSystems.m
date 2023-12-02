function [d_x, d_lambda, d_nu] = SolvePDKKTSystems(...
    x, lambda, nu, grad_f, Hessian_f, m, G, diff_G, Hessian_G, A, b, t)
% SolveBarrierKKTSystems

% Jingyu Liu, December 2, 2023.

% Solve the primal-dual interior-point KKT systems
% at point x.

% We don't recommend use this default solver.

arguments (Input)
    x (:, 1) double;
    lambda (:, 1) double;
    nu (:, 1) double;
    grad_f function_handle;
    Hessian_f function_handle;
    m (1, 1) double
    G function_handle;
    diff_G function_handle;
    Hessian_G function_handle;
    A (:, :) double;
    b (:, 1) double;
    t (:, 1) double;
end

arguments (Output)
    d_x (:, 1) double;
    d_lambda (:, 1) double;
    d_nu (:, 1) double;
end

G_x = G(x);
diff_G_x = diff_G(x);
Hessian_G_x = Hessian_G(x);
H = Hessian_f(x);
gg = grad_f(x) - sum(G_x .\ diff_G_x)' / t + A' * nu;
h = A * x - b;

for i = 1 : m
    H = H ...
        + lambda(i) * Hessian_G_x(:, :, i) ...
        - lambda(i) * diff_G_x(i, :)' * diff_G_x(i, :) / G_x(i);
end

[p, n] = size(A);

HinvAtg = H \ [A', gg];
HinvAt = HinvAtg(:, 1 : p);
Hinvg = HinvAtg(:, p + 1);
S = -A * HinvAt;
d_nu = S \ (A * Hinvg - h);
d_x = H \ (-A' * d_nu - gg);
r_cent = -lambda .* G_x - ones(m, 1) / t;
d_lambda = G_x .\ (-lambda .* diff_G_x * d_x + r_cent);

end