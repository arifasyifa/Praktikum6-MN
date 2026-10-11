clc; clear; close all;

A = [ 5  -1  -1   0   0;
     -1   5  -1  -1   0;
     -1  -1   4  -1  -1;
      0   0   1   4  -2;
      0   1  -1   1   4 ];
b = [-1; 2; 6; 2; -1];

maxIter = 20; tol = 0.001;

% --- Gauss ---
tic;
x_gauss = A\b;
t_gauss = toc;

% --- Gauss-Jordan ---
tic;
A_aug = [A b];
n = size(A,1);
for i=1:n
    A_aug(i,:) = A_aug(i,:) / A_aug(i,i);
    for j=1:n
        if j~=i
            A_aug(j,:) = A_aug(j,:) - A_aug(i,:)*A_aug(j,i);
        end
    end
end
x_gj = A_aug(:,end);
t_gj = toc;

% --- LU ---
tic;
[L,U] = lu(A);
y = L\b;
x_lu = U\y;
t_lu = toc;

% --- Jacobi ---
tic;
x_jac = zeros(5,1);
for k = 1:maxIter
    x_old = x_jac;
    for i = 1:5
        s = 0;
        for j = 1:5
            if j ~= i
                s = s + A(i,j)*x_old(j);
            end
        end
        x_jac(i) = (b(i) - s)/A(i,i);
    end
    if norm(x_jac - x_old, inf) < tol
        break;
    end
end
t_jac = toc;

% --- Gauss-Seidel ---
tic;
x_gs = zeros(5,1);
for k = 1:maxIter
    x_old = x_gs;
    for i = 1:5
        s1 = A(i,1:i-1)*x_gs(1:i-1);
        s2 = A(i,i+1:end)*x_gs(i+1:end);
        x_gs(i) = (b(i) - s1 - s2)/A(i,i);
    end
    if norm(x_gs - x_old, inf) < tol
        break;
    end
end
t_gs = toc;

fprintf('\nWaktu Komputasi:\n');
fprintf('Gauss: %.6f detik\n', t_gauss);
fprintf('Gauss-Jordan: %.6f detik\n', t_gj);
fprintf('LU Decomposition: %.6f detik\n', t_lu);
fprintf('Jacobi: %.6f detik\n', t_jac);
fprintf('Gauss-Seidel: %.6f detik\n', t_gs);

