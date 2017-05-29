clear all;
K = [1545.0966799187809, 0, 639.5; ...
     0, 1545.0966799187809, 359.5; ...
     0, 0, 1];

% read the data
point2DOrig = readPoint('hw3_points2D.txt', 2);
point3DOrig = readPoint('hw3_points3D.txt', 3);

% transfer to homogeneous
point2DHomo = [point2DOrig, ones(60, 1)];
point3DHomo = [point3DOrig, ones(60, 1)];

% calculate the point in normalized coordinates
point2DNorm = (K \ (point2DHomo'))';



% question (a) %
[inlierIndex, trials] = MSAC(K, point2DOrig, point3DOrig);
disp('Question (a):');
disp('number of trials:');
disp(trials);
num_inliers = sum(inlierIndex);
disp('number of inliers');
disp(num_inliers);



% question (b) %

% pick up inliers
inlier2DNorm = point2DNorm(inlierIndex, :);
inlier3DInhomo = point3DOrig(inlierIndex, :);

% linear estimation
[R, t, flag] = linearEst(inlier2DNorm, inlier3DInhomo);
P_linear = [R, t];

format longg;
disp('Question (b):')
disp('camera projection matrix [R|t]:');
disp(P_linear);



% question (c) %

% pick up inliers
inlier2DOrig = point2DOrig(inlierIndex, :)';
inlier3DOrig = point3DOrig(inlierIndex, :)';
inlier2DHomo = point2DHomo(inlierIndex, :)';
inlier3DHomo = point3DHomo(inlierIndex, :)';
inlier2DNorm = point2DNorm(inlierIndex, :)';
inlier2DNorm = inlier2DNorm(1 : 2, :);

[P, w] = levenberg(P_linear, K, inlier2DOrig, inlier3DHomo);
disp('Question (c):')
disp('final result of w:');
disp(w);
disp('final matrix [R|t]:');
disp(P);


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