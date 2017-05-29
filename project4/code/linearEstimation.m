% DLT linear estimation %
function [H_norm, T1, T2] = linearEstimation(inlier1, inlier2)
    % data normalization
    [point1, T1] = dataNormalization(inlier1, 2);
    [point2, T2] = dataNormalization(inlier2, 2);

    % using DLT compute matrix A
    % H * point2 = point1
    A = DLTAlgorithm(point1, point2);

    % using svd compute projection matrix P
    H_norm = calProjMat(A);
end

% Data Normalization
function [point, T] = dataNormalization(data, dim)
    num_point = size(data, 1);
    % calculate mean and variance
    m = mean(data);
    v = var(data);
    % calculate normalized vector
    if (dim == 2)
        s = sqrt(2.0 / sum(v));
        T = zeros(3, 3);
        T(1, 1) = s;
        T(2, 2) = s;
        T(3, 3) = 1.0;
        T(1, 3) = -1.0 * m(1) * s;
        T(2, 3) = -1.0 * m(2) * s;
    else
        s = sqrt(3.0 / sum(v));
        T = zeros(4, 4);
        T(1, 1) = s;
        T(2, 2) = s;
        T(3, 3) = s;
        T(4, 4) = 1.0;
        T(1, 4) = -1.0 * m(1) * s;
        T(2, 4) = -1.0 * m(2) * s;
        T(3, 4) = -1.0 * m(3) * s;
    end
    % transfer inhomo to homo data
    unit = ones(num_point, 1);
    point = [data, unit];
    % normalize the data
    point = T * point';
    point = point';
end

% DLT algorithm
function matA = DLTAlgorithm(point1, point2)
    num_point = size(point1, 1);
    point1 = point1';
    point2 = point2';
    matA = [];
    % using house holder matrix
    % to calculate left null space of x
    for i = 1 : num_point
        x = point1(:, i);
        v = x + sign(x(1)) * norm(x) * [1, 0, 0]';
        Hv = eye(3) - 2.0 * (v * v') / (v' * v);
        leftNull = Hv(2 : 3, :);
        matA = [matA; kron(leftNull, point2(:, i)')];
    end
end

% using svd compute projection matrix P
function P = calProjMat(A)
    [U, S, V] = svd(A);
    P = V(:, size(V, 2));
    P = reshape(P, [3, 3]);
    P = P';
end