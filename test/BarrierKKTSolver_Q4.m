function [Newton_step_x, w, lambda_square] = BarrierKKTSolver_Q4(...
    x, c, A, b, t)
% BarrierKKTSolver_Q4

% Jingyu Liu, December 1, 2023.

arguments (Input)
    x (:, 1) double;
    c (:, 1) double;
    A (:, :) double;
    b (:, 1) double;
    t (:, 1) double;
end

arguments (Output)
    Newton_step_x (:, 1) double;
    w (:, 1) double;
    lambda_square (1, 1) double;
end

% psi = diagonal elements of Hinv.
deno = (2 * x.^2 - 2 * x + 1);
nume = (1 - x) .* x;
psi = nume.^2 ./ deno;
Hinv_c = psi .* c;
Hinv_grad_phi = ((2 * x - 1) .* nume) ./ deno;
Hinv_g = t * Hinv_c + Hinv_grad_phi;

w = (A * (psi .* A')) \ (A * (x - Hinv_g) - b);
Newton_step_x = -Hinv_g - psi .* A' * w;
lambda_square = Newton_step_x' * (psi .\ Newton_step_x);

end