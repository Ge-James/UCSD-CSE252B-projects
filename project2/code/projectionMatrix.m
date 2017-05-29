% Question (a) %
% Linear Estimation %

% DLT algorithm to estimate camera projection matrix

% read the data
point2Dorig = readPoint('hw2_points2D.txt', 2);
point3Dorig = readPoint('hw2_points3D.txt', 3);

% data normalization
[point2D, T] = dataNormalization(point2Dorig, 2);
[point3D, U] = dataNormalization(point3Dorig, 3);

% using DLT compute matrix A
A = DLTAlgorithm(point2D, point3D);

% using svd compute projection matrix P
P_norm = calProjMat(A);
% P_norm = - P_norm;

% scale P with ||P||Fro = 1
P = inv(T) * P_norm * U;
P = P / norm(P, 'fro');
format shortg;
disp(P);

uni = ones(50, 1);
xEst = P * [point3Dorig, uni]';
paramW = xEst(3, :);
xEst = xEst ./ paramW;
% disp(xEst);


% Problem 2%
% Iterative Method %

% use original data instead of normalized
unit = ones(50, 1);
x2D = [point2Dorig, unit]';
x3D = [point3Dorig, unit]';

% data normalization
[x2D_norm, T] = dataNormalization(point2Dorig, 2);
[x3D_norm, U] = dataNormalization(point3Dorig, 3);

x2D = x2D_norm;
x3D = x3D_norm;

% main function
format long;
lambda = 0.001;
measureVector = calMeasureVector(x2D);
% covar = calCovar(x2D, T);
covar = eye(100);
covar = covar * T(1, 1)^2;
% deparamP = P_norm;
deparamP = -[P_norm(1, :), P_norm(2, :), P_norm(3, :)]';
paramP = parameterize(deparamP);
% compute error and cost
proj2D = projection(deparamP, x3D);
error = measureVector - calMeasureVector(proj2D);
cost = error' * inv(covar) * error;
disp(cost);
% begin iteration
for i = 1 : 30
    % compute Jocabian
    J = calJocabian(x2D, x3D, deparamP, paramP);
    % compute delta
    delta = (J' * inv(covar) * J + lambda * eye(11)) \ (J' * inv(covar) * error);
    % update P
    paramPUpdate = paramP + delta;
    deparamPUpdate = deparameterize(paramPUpdate);
    % compute error and cost
    proj2DUpdate = projection(deparamPUpdate, x3D);
    errorUpdate = measureVector - calMeasureVector(proj2DUpdate);
    costUpdate = errorUpdate' * inv(covar) * errorUpdate;
    disp(costUpdate);
    % make decsion
    if (costUpdate < cost)
        paramP = paramPUpdate;
        deparamP = deparamPUpdate;
        error = errorUpdate;
        cost = costUpdate;
        lambda = 0.1 * lambda;
    else
        lambda = 10.0 * lambda;
    end
end

proMat_norm = reshape(deparamP, [4, 3])';
proMat = inv(T) * proMat_norm * U;
proMat = proMat / norm(proMat, 'fro');
format shortg;
disp(proMat);



% read the data
function point = readPoint(fileName, dim)
    file = fopen(fileName);
    if dim == 2
        point = textscan(file, '%f %f');
    else
        point = textscan(file, '%f %f %f');
    end
    fclose(file);
    point = cell2mat(point);
end

% Data Normalization
function [point, T] = dataNormalization(data, dim)
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
    unit = ones(50,1);
    point = [data, unit]';
    % normalize the data
    point = T * point;
end

% DLT algorithm
function matA = DLTAlgorithm(point2D, point3D)
    matA = [];
    % using house holder matrix
    % to calculate left null space of x
    for i = 1 : 50
        x = point2D(:, i);
        v = x + sign(x(1)) * norm(x) * [1, 0, 0]';
        Hv = eye(3) - 2.0 * (v * v') / (v' * v);
        leftNull = Hv(2:3, :);
        matA = [matA; kron(leftNull, point3D(:, i)')];
    end
end

% using svd compute projection matrix P
function P = calProjMat(A)
    [U, S, V] = svd(A);
    P = V(:, 12);
    P = reshape(P, [4, 3]);
    P = P';
end



% Question (b)%
% construct measurement vector
function measureVector = calMeasureVector(point)
    measureVector = [];
    for i = 1 : 50
        measureVector = [measureVector, point(1 : 2, i)'];
    end
    measureVector = measureVector';
end

% projection from 3D to 2D
function  res = projection(deparamP, x3D)
    P = reshape(deparamP, [4, 3])';
    res = P * x3D;
    w = res(3, :);
    res = res ./ w;
end

% construct associated covariance
function covar = calCovar(point, T)
    covar = eye(100);
%     for i = 1 : 50
%         covar(2 * i - 1: 2 * i, 2 * i - 1 : 2 * i) = cov(point(1 : 2, i)', point(1 : 2, i)');
%     end
    covar = T(1, 1) * T(1, 1) * covar;
end

% parameterize
function paramVector = parameterize(P)
    a = P(1);
    b = P(2 : length(P));
    paramVector = (2.0 / (sinc(acos(a)))) * b;
    normP = norm(paramVector);
    if (normP > pi)
        paramVector = (1.0 - 2 * pi / normP * ceil((normP - pi) / 2 * pi)) * paramVector;
        % paramVector = (1.0 - 2 * pi / normP) * paramVector;
    end
end

% deparameterize
function deparamVector = deparameterize(P)
    normP = norm(P);
    deparamVector = [cos(normP / 2.0), ((sinc(normP / 2.0)) / 2.0) * P']';
end

% sinc(x)
function res = sinc(x)
    if x == 0
        res = 1.0;
    else
        res = (sin(x)) / x;
    end
end

% compute Jocabian
% x2D and x3D are homogeneous
function jocabian = calJocabian(x2D, x3D, deparamP, paramP)
    jocabian = [];
    projX2D = projection(deparamP, x3D);
    part2 = jocab2(deparamP, paramP);
    for i = 1: 50
        part1 = jocab1(projX2D(:, i), x3D(:, i), deparamP);
        jocabian = [jocabian; part1 * part2];
    end
end

% compute partial xi partial P bar
function res = jocab1(point2D, point3D, deparamP)
    w = deparamP(9 : 12)' * point3D;
    tmp = zeros(1, 4);
    res = 1 / w * [point3D', tmp, -1.0 * point2D(1) * point3D'; ...
                   tmp, point3D', -1.0 * point2D(2) * point3D'];
end

% compute partial P bar partial P
function res = jocab2(deparamP, paramP)
    normP = norm(paramP);
    res = -0.5 * deparamP(2 : length(deparamP))';
    if (normP == 0)
        res = [res; 0.5 * eye(length(paramP))];
    else
        tmp = 0.5 * (sinc(normP / 2)) * eye(length(paramP)) + 0.25 / normP ...
              * derivSinc(normP / 2) * paramP * paramP';
        res = [res; tmp];
    end
end

% derivative of sinc(x)
function res = derivSinc(x)
    if x == 0
        res = 0.0;
    else
        res = cos(x) / x - sin(x) / (x * x);
    end
end