function Nomer1()
clc; clear; close all;

% Matriks koefisien (A) dan vektor konstanta (b)
A = [ 5  -1  -1   0   0;
     -1   5  -1  -1   0;
     -1  -1   4  -1  -1;
      0   0   1   4  -2;
      0   1  -1   1   4 ];

b = [-1; 2; 6; 2; -1];

% === Metode 1: Gauss Elimination (langsung) ===
x_gauss = A\b;
disp('Solusi metode Gauss (langsung):');
disp(x_gauss);

% === Metode 2: Gauss-Jordan ===
x_gj = metodeGaussJordan(A, b);
disp('Solusi metode Gauss-Jordan:');
disp(x_gj);

% === Metode 3: Decomposisi LU ===
x_lu = metodeLU(A, b);
disp('Solusi metode Decomposisi LU:');
disp(x_lu);

% === Metode 4: Iterasi Jacobi ===
x_jac = metodeJacobi(A, b, 20, 0.001);
disp('Solusi metode Jacobi:');
disp(x_jac);

% === Metode 5: Iterasi Gauss-Seidel ===
x_gs = metodeGaussSeidel(A, b, 20, 0.001);
disp('Solusi metode Gauss-Seidel:');
disp(x_gs);

end % akhir fungsi utama

function x = metodeGaussJordan(A, b)
    A_aug = [A b];
    n = size(A,1);
    for i=1:n
        A_aug(i,:) = A_aug(i,:) / A_aug(i,i);
        for j=1:n
            if j ~= i
                A_aug(j,:) = A_aug(j,:) - A_aug(i,:) * A_aug(j,i);
            end
        end
    end
    x = A_aug(:,end);
end

function x = metodeLU(A, b)
    [L, U] = lu(A);
    y = L\b;
    x = U\y;
end
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

function x = metodeGaussSeidel(A, b, maxIter, tol)
    n = length(b);
    x = zeros(n,1);
    for k = 1:maxIter
        x_old = x;
        for i = 1:n
            s1 = A(i,1:i-1)*x(1:i-1);
            s2 = A(i,i+1:end)*x(i+1:end);
            x(i) = (b(i) - s1 - s2)/A(i,i);
        end
        if norm(x - x_old, inf) < tol
            break;
        end
    end
end
