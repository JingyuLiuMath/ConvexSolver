function [step_size, new_x, f_new_x] = BacktrackingLineSearch( ...
    f, x, grad_f_x, dir_x, ...
    alpha, beta)
% BacktrackingLineSearch

% Jingyu Liu, December 1, 2023.

arguments (Input)
    f function_handle;
    x (:, 1) double;
    grad_f_x (:, 1) double;
    dir_x (:, 1) double;
    alpha (1, 1) double = 0.1;
    beta (1, 1) double = 0.5;
end

arguments (Output)
    step_size (1, 1) double;
    new_x (:, 1) double;
    f_new_x (1, 1) double;
end

f_x = f(x);
descent_value = grad_f_x' * dir_x;  % descent_value < 0.
step_size = 1;
new_x = x + step_size * dir_x;
f_new_x = f(new_x);
while f_new_x > f_x + alpha * step_size * descent_value
    step_size = beta * step_size;
    new_x = x + step_size * dir_x;
    f_new_x = f(new_x);
end

end