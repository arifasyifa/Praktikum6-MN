function Nomer2()
clc; clear; close all;

% Data SPL
A = [ 5  -1  -1   0   0;
     -1   5  -1  -1   0;
     -1  -1   4  -1  -1;
      0   0   1   4  -2;
      0   1  -1   1   4 ];
b = [-1; 2; 6; 2; -1];

x_exact = A\b;
maxIter = 20;
tol = 0.001;

% Panggil fungsi iteratif (subfunctions ada di bawah)
x_jac = metodeJacobi(A, b, maxIter, tol);
x_gs  = metodeGaussSeidel(A, b, maxIter, tol);

% Hitung galat absolut dan bulatkan ke 6 desimal
galat_jac = round(abs(x_exact - x_jac) * 1e6) / 1e6;
galat_gs  = round(abs(x_exact - x_gs)  * 1e6) / 1e6;

% Tampilkan hasil tabel
fprintf('NOMER 2\n');
fprintf('%12s %15s\n','Jacobi','Gauss-Seidel');
for i = 1:length(b)
    fprintf('%12.6f %15.6f\n', galat_jac(i), galat_gs(i));
end

end

% Subfunction: Jacobi
function x = metodeJacobi(A, b, maxIter, tol)
    n = length(b);
    x = zeros(n,1);
    for k = 1:maxIter
        x_old = x;
        for i = 1:n
            s = 0;
            for j = 1:n
                if j ~= i
                    s = s + A(i,j)*x_old(j);
                end
            end
            x(i) = (b(i) - s)/A(i,i);
        end
        if norm(x - x_old, inf) < tol
            break;
        end
    end
end

% Subfunction: Gauss-Seidel
function x = metodeGaussSeidel(A, b, maxIter, tol)
    n = length(b);
    x = zeros(n,1);
    for k = 1:maxIter
        x_old = x;
        for i = 1:n
            s1 = 0;
            for j = 1:i-1
                s1 = s1 + A(i,j)*x(j);
            end
            s2 = 0;
            for j = i+1:n
                s2 = s2 + A(i,j)*x(j);
            end
            x(i) = (b(i) - s1 - s2)/A(i,i);
        end
        if norm(x - x_old, inf) < tol
            break;
        end
    end
end

