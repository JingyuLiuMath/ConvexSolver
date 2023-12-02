function [Newton_step_x, w, lambda_square] = KKTSolver_Q3(x, A, b)
% KKTSolver_Q3

% Jingyu Liu, December 1, 2023.

arguments (Input)
    x (:, 1) double;
    A (:, :) double;
    b (:, 1) double;
end

arguments (Output)
    Newton_step_x (:, 1) double;
    w (:, 1) double;
    lambda_square (1, 1) double;
end

x_square = x.^2;
w = (A * (x_square .* A')) \ (2 * A * x - b);
Newton_step_x = x - x_square .* (A' * w);
lambda_square = Newton_step_x' * (x_square .\ Newton_step_x);

end