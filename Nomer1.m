clc; clear;

A = [10 -1 2 0 1; -1 11 -1 3 0; 2 -1 10 -1 1; 0 3 -1 8 -1; 1 0 1 -1 9];  % GANTI dengan matriks A kamu
b = [6; 25; -11; 15; 8];   % GANTI dengan vektor b kamu
n = length(b);
maxIter = 2;
x0 = zeros(n,1);

% 1. Gauss
x_gauss = A\b;

% 2. Gauss-Jordan
Aug = [A, b];
for i = 1:n
  Aug(i,:) = Aug(i,:) / Aug(i,i);
  for j = 1:n
    if j ~= i
      Aug(j,:) = Aug(j,:) - Aug(j,i) * Aug(i,:);
    end
  end
end
x_gj = Aug(:,n+1);

% 3. Dekomposisi LU
[L, U, P] = lu(A);
y = L \ (P*b);
x_lu = U \ y;

% 4. Jacobi (2 iterasi)
x_jac = x0;
for k = 1:maxIter
  x_new = x_jac;
  for i = 1:n
    s = A(i,:) * x_jac - A(i,i) * x_jac(i);
    x_new(i) = (b(i) - s) / A(i,i);
  end
  x_jac = x_new;
end

% 5. Gauss-Seidel (2 iterasi)
x_gs = x0;
for k = 1:maxIter
  for i = 1:n
    s = A(i,:) * x_gs - A(i,i) * x_gs(i);
    x_gs(i) = (b(i) - s) / A(i,i);
  end
end

disp('Solusi Gauss');        disp(x_gauss);
disp('Solusi Gauss-Jordan'); disp(x_gj);
disp('Solusi LU');           disp(x_lu);
disp('Solusi Jacobi');       disp(x_jac);
disp('Solusi Gauss-Seidel'); disp(x_gs);
